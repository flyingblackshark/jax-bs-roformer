import tensorflow as tf
import librosa
import os
from array_record.python.array_record_module import ArrayRecordWriter


if __name__ == "__main__":
    folder_path = "/mnt/e/musdb18hq/train/"
    result = []
    for root, _, files in os.walk(folder_path):
        if 'mixture.wav' in files and 'vocals.wav' in files:
            result.append(root)
    i = 0
    writer = None
    for item in result:
        if i%10240 == 0:
            print(f"round {i+1}")
            num = i//10240
            if writer is not None:
                writer.close() 
            writer = ArrayRecordWriter(f"./dataset/bsr_dataset_part_{num}.arrayrecord", 'group_size:1')
        mixture_path = os.path.join(item, 'mixture.wav')
        vocals_path = os.path.join(item, 'vocals.wav')

        # 使用 librosa 读取音频文件
        try:
            mixture_audio, mixture_sr = librosa.load(mixture_path, sr=44100,mono=False)
            vocals_audio, vocals_sr = librosa.load(vocals_path, sr=44100,mono=False)

        except Exception as e:
            print(f"Error reading audio files in {root}: {e}")
        example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'mixture': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(mixture_audio).numpy()])),
                        'vocals':tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(vocals_audio).numpy()])
                        ),
                    }
                )
            )
        writer.write(example.SerializeToString())
        i+=1
    writer.close() 
    