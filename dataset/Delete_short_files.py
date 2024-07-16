import os
import wave

def delete_short_wav_files(directory):
    total_deleted = 0
    deleted_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    with wave.open(file_path, 'rb') as wav_file:
                        frame_rate = wav_file.getframerate()
                        n_frames = wav_file.getnframes()
                        duration = n_frames / float(frame_rate)

                        if frame_rate == 48000 and duration < 1.0:
                            os.remove(file_path)
                            deleted_files.append(file_path)
                            total_deleted += 1
                except wave.Error as e:
                    print(f"Error processing file {file_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error occurred while processing file {file_path}: {e}")

    print(f"Total files deleted: {total_deleted}")
    for file in deleted_files:
        print(f"Deleted file: {file}")

# 使用示例
directory = '/home/SE_dataset/VCTK_Demand_ch1/Train_clean'
delete_short_wav_files(directory)
