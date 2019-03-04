import pandas as pd
import os
import glob
import shutil

def copy_files(csv_file, dir_audio, result_dir):
    wrong_fnames = []
    if not check_exist(csv_file, result_dir):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        df = pd.read_csv(csv_file, header=0, sep="\t")
        df_filenames = df["filename"]
        print("nb of meta : " + str(len(df_filenames)))
        for filename in df_filenames:
            audio_file = os.path.join(dir_audio, filename)
            result_file = os.path.join(result_dir, filename)
            try:
                if not os.path.exists(result_file):
                    shutil.copy(audio_file, result_file)
                else:
                    print(filename + " already exist")
            except Exception as e:
                print(str(e))
                wrong_fnames.append(filename)
        print("nb of files copied :" + str(len(glob.glob(os.path.join(result_dir, "*")))))
    return wrong_fnames



def check_exist(csv_file, result_dir):
    print(csv_file)
    if os.path.exists(result_dir):
        existing_files = glob.glob(os.path.join(result_dir, "*"))
        existing_files = pd.Series([fname.split('/')[-1] for fname in existing_files])
        meta_files = pd.read_csv(csv_file, header=0, sep="\t")["filename"]
        
        missing_audios = meta_files[~meta_files.isin(existing_files)]
        exceed_audios = existing_files[~existing_files.isin(meta_files)]
        
        if not exceed_audios.empty:
            print("nb of exceeding files: " + str(len(exceed_audios)))
            print("removing ...")
            for audio_file in exceed_audios:
                os.remove(os.path.join(result_dir, audio_file))
        if missing_audios.empty:
            print(csv_file + " match the result dir: " + result_dir)
            return True
        else:
            print("nb of missing files: " + str(len(missing_audios)))   
    return False


if __name__ == "__main__":
    # change with the directory pass where missing_file*.csv are.
    dir_missing_files = "/home/nturpault/code/task_4dcase2018/dataset"
    
    dir_dataset = "/talc3/multispeech/calcul/users/rserizel/DCASE2018/task4"


    missing_test = os.path.join(dir_missing_files, "missing_files_test.csv")
    missing_weak = os.path.join(dir_missing_files, "missing_files_weak.csv")
    missing_unlabel_in_domain = os.path.join(dir_missing_files, "missing_files_unlabel_in_domain.csv")
    missing_unlabel_out_of_domain = os.path.join(dir_missing_files, "missing_files_unlabel_out_of_domain.csv")

    for missing_train in [missing_weak, missing_unlabel_in_domain, missing_unlabel_out_of_domain]:
        if os.path.exists(missing_train):
            final_dir = os.path.join("audio", "train", missing_train.split('missing_files_')[-1][:-4])
            
            audio_dir = os.path.join(dir_dataset, "dataset", final_dir)
            result_dir = os.path.join("missing_files", final_dir)

            wrongs = copy_files(missing_train, audio_dir, result_dir)
            print("wrong files: " + str(wrongs))
        else:
            print(missing_train + " does not exist")
    
    if os.path.exists(missing_test):
        final_dir = os.path.join("audio", "test")
        audio_test = os.path.join(dir_dataset, "dataset", final_dir)
        result_test = os.path.join("missing_files", final_dir)
        
        wrongs = copy_files(missing_test, audio_test, result_test)
        print("wrong files: " + str(wrongs))
    else:
        print(missing_test + " does not exist")

    if os.path.exists(missing_eval):
        final_dir = os.path.join("audio", "eval")
        audio_eval = os.path.join(dir_dataset, "dataset", final_dir)
        result_eval = os.path.join("missing_files", final_dir)
        
        wrongs = copy_files(missing_eval, audio_eval, result_eval, sep=sep)
            print("wrong files: " + str(wrongs))
    else:
        result_eval = os.path.join("missing_files", "audio", "eval")
        if os.path.exists(result_eval):
            shutil.rmtree(result_eval)
            print("removing eval directory ...")
        print(missing_eval + " does not exist")
