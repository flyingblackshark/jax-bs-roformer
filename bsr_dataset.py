import dataclasses
import glob
import tensorflow as tf
import jax
import grain.python as grain
import multihost_dataloading
import numpy as np
@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
    """Parse serialized example"""

    def __init__(self, data_columns ,dtype=tf.float32):
        self.data_columns = data_columns
        self.dtype = dtype

    def map(self, features):
        def _parse(example):
            # Dynamically create the feature description dictionary
            feature_description = {
                column: tf.io.FixedLenFeature([], dtype=tf.string)
                for column in self.data_columns
            }
            parsed = tf.io.parse_example(example, feature_description)

            parsed = {
                key: tf.io.parse_tensor(value, self.dtype).numpy()
                for key, value in parsed.items()
            }
            return parsed

        return _parse(features)
  
@dataclasses.dataclass
class SliceToLength(grain.RandomMapTransform):
    def __init__(self,data_columns,segment_size):
        self.segment_size = segment_size
        self.data_columns = data_columns
    def random_map(self, data, rng: np.random.Generator):
        idx = rng.integers(0, data[self.data_columns[0]].shape[-1] - self.segment_size - 2)
        # 按照 data_columns 遍历并切片
        for column in self.data_columns:
            if column in data:
                data[column] = data[column][...,idx:idx + self.segment_size]
            else:
                raise KeyError(f"Column '{column}' not found in data.")
        
        return data
    
@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
    """Pads each input to the specified length"""

    def __init__(self, max_length):
        self.max_length = max_length

    def map(self, data):
        """map to each element"""

        def _pad(x, max_length):
            pad_amount = max(max_length - x.shape[0], 0)
            pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
            return np.pad(x, pad_amount)

        for key, _ in data.items():
            data[key] = _pad(data[key], self.max_length)
        return data
  
def get_datasets(data_file_pattern):
  """Load dataset from array_record files for using with grain"""
  data_files = glob.glob(data_file_pattern)
  dataset = grain.ArrayRecordDataSource(data_files)
  return dataset

def preprocessing_pipeline(
    dataset,
    global_batch_size: int,
    global_mesh,
    segment_length: int,
    grain_worker_count: int,
    dataloading_host_index,
    dataloading_host_count,
    data_columns,
    shuffle: bool = False,
    data_shuffle_seed=0,
    num_epochs=1,
    drop_remainder=False,
):
    """Use grain to pre-process the dataset and return iterators"""
    assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

    operations = []
    operations.append(ParseFeatures(data_columns))
    operations.append(SliceToLength(data_columns,segment_length))
    #operations.append(PadToMaxLength(data_column,max_target_length))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder))


    index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(
            shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=drop_remainder
        ),
        shuffle=shuffle,
        seed=data_shuffle_seed,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=index_sampler,
        worker_count=grain_worker_count,
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

    # Return multi-host jax.Array prep iterator
    return multihost_gen