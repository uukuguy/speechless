import pickle


class PickleUtils:

    @staticmethod
    def load_pickle(filepath):
        res = None
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
        return res

    @staticmethod
    def save_pickle(res, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(res, f, protocol=4)