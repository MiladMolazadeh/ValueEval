from configuration import BaseConfig
import pandas as pd
from sklearn.metrics import f1_score
import json
if __name__ == '__main__':
    CONFIG = BaseConfig().get_config()
    target_df = pd.read_csv(CONFIG.train_labels_path,
                            encoding='utf-8', sep='\t', header=0)
    predict_df = pd.read_csv("../results/result-train.tsv",
                            encoding='utf-8', sep='\t', header=0)

    targets = json.loads(target_df.to_json(orient='records'))
    predicts = json.loads(predict_df.to_json(orient='records'))

    print()
    f1 = 0
    i=0
    for target, predict in zip(targets, predicts):
        labels, tid = list(target.values())[1:], list(target.values())[0]
        preds, pid = list(predict.values())[1:], list(predict.values())[0]
        f1 += f1_score(preds, labels)
        assert tid == pid
        i+=1
    print(f1/i)
    print()



