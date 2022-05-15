import argparse
import sklearn.datasets

import minizinc
import numpy as np



def train(params):

    X, y = load_dataset(params.images)


    dict_const = {
        "ROWS": X.shape[1],
        "COLUMNS": X.shape[2],
        "IMAGES": params.images,
        "labels": list(y),
        "int_images": list(X)
    }

    result = run_optimization("src/minizinc/fit.mzn", dict_const)

    print(result.objective  / X.shape[0])


def run_optimization(mzn_file, dict_const={}, solver="gecode", timeout=None, processes=1):
    problem = minizinc.Model(mzn_file)
    solver = minizinc.Solver.lookup(solver)

    instance = minizinc.Instance(solver, problem)

    for key, value in dict_const.items():
        instance[key] = value

    if timeout is not None:
        timeout = datetime.timedelta(seconds=timeout)

    results = instance.solve(timeout=timeout, processes=processes)

    return results


def load_dataset(number):
    arr_data, arr_targets = sklearn.datasets.load_digits(n_class=2, return_X_y=True)

    return arr_data[:number].astype(int).reshape((-1, 8,8)),   arr_targets[:number] > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=int,
                          help='number of images')
    parser.add_argument("-v", "--verbose", action="store_true",
                          help='tensorboard log directory')
    params = parser.parse_args()


    train(params)

if __name__ == '__main__':
    main()














