import os
import pandas as pd
import pickle


def create_dir(dirname):
    # path = os.getcwd() + '/' + dirname
    path = dirname
    folder = os.path.exists(path)

    try:
        if not folder:
            os.makedirs(path, exist_ok=True)
    except OSError as err:
        print(err)

    return path + "/"


def save_results(save_path, score):
    # with open(fres + project_name + '-' + model_name + '.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(score)

    # pandas
    tempRes = pd.DataFrame(score).T
    tempRes.to_csv(save_path + '.csv', index=False, header=False, mode='a')


def save_results_pickle(save_path, results):
    with open(save_path + '.pkl', 'ab') as f:
        pickle.dump(results, f)
    f.close()


def load_results_pickle(save_path):
    results = []
    with open(save_path + '.pkl', 'rb') as f:
        result = pickle.load(f)
        if type(result) == list:
            results = result
        else:
            results.append(result)
            while True:
                try:
                    result = pickle.load(f)
                    results.append(result)
                except EOFError:
                    break
    f.close()

    return results
