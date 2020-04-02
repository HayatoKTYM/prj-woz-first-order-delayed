"""
-object
行動生成の定量的評価

-detail
u(t) = 1 の 間で ,
modelの出力が閾値を超えた回数 (予測値)
実際にsystemが行動した回数(真値)
から、 precision-recall-F1 を計算
"""

import os


def quantitative_evaluation(
                    y_true, 
                    y_pred,
                    u,
                    threshold=0.8,
                    resume=False,
                    output='./'
                    ):
    target = False
    pred = False
    flag = True
    Distance = []
    a, b, c = 0, 0, 0
    u_t_count = 0

    for i in range(len(y_true)-1):
        #  発話中 : 評価対象外
        if u[i] == 0:
            target = False
            pred = False
            u_t_count = 0
        #  u(t) = 1 : u_t_count が 閾値を超えたら終わりにする用
        else:
            u_t_count += 1
        #  予測が閾値を超えたタイミング
        if y_pred[i] >= threshold and flag:
            pred = True
            flag = False
            pred_frame = i
        #  正解ラベルのタイミング
        if y_true[i] > 0:
            target = True
            target_frame = i
        #  u_t が 1→0 に変わるタイミング or u(t)=1 が 一定以上続いた時
        if (u[i] == 1 and u[i+1] == 0 ):
            flag = True
            u_t_count = 0
            if pred and target:
                a += 1
                Distance.append(pred_frame-target_frame)
            elif pred:
                b += 1
            elif target:
                c += 1

    print(a, b, c)
    precision = a / (a + b)
    recall = a / (a + c)
    f1 = precision * recall * 2 / (precision + recall)
    print('precision is {}, recall is {}, f1 is {}'.format(precision, recall, f1))
    if resume:
        fo = open(os.path.join(output, 'eval_report.txt'),'a')
        print("""
            precision is {}, recall is {}, f1 is {}
            """.format(precision, recall, f1), file=fo)
        fo.close()

    return precision, recall, f1, Distance
    