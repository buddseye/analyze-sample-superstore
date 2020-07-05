# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier as DT
clf = DT(max_depth=3, min_samples_leaf=500)


def sub(x, y):
    '''Simple subtract function.
    >>> sub([4, 3], [2, 1])
    [2, 2]
    '''
    return np.subtract(x, y).tolist()


def clustering(x, y):
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([x, y])
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=1, min_samples=3).fit(X)
    return db.labels_.tolist()


def _desision_tree():
    import pandas as pd
    df1 = pd.read_excel('SampleSuperStore.xls', sheet_name='注文')
    df2 = pd.read_excel('SampleSuperStore.xls', sheet_name='返品')
    df = pd.merge(df1, df2, on='オーダー ID', how='left')
    df['返品'] = df['返品'].fillna('×')
    # 目的変数
    y = df['返品']
    # 説明変数
    train_x = df[df['オーダー日'].dt.year < 2019]
    train_x = df[['出荷モード', '顧客区分', '地域', 'カテゴリ', '数量', '割引率']]
    # ダミー変数化
    train_x = pd.get_dummies(train_x)
    # モデル作成
    clf.fit(train_x, y)


def decision_tree(mode, customer_category, region, category, quantity, ratio):
    import pandas as pd
    # 入力変数をデータフレームとして格納
    df_mode = pd.DataFrame(mode)
    df_customer_category = pd.DataFrame(customer_category)
    df_region = pd.DataFrame(region)
    df_category = pd.DataFrame(category)
    df_quantity = pd.DataFrame(quantity)
    df_ratio = pd.DataFrame(ratio)

    # 入力変数を１つのデータフレームに連結
    df = pd.concat([
        df_mode,
        df_customer_category,
        df_region,
        df_category,
        df_quantity,
        df_ratio
    ], axis=1)
    print(df)

    df_dummy = pd.get_dummies(df)

    # 作成したモデルを利用して予測値を取得
    y_pred = clf.predict_proba(df_dummy)

    # 予測結果をリストとして返す
    return y_pred[:, 1].tolist()


if __name__ == '__main__':
    from tabpy.tabpy_tools.client import Client
    client = Client('http://localhost:9004/')
    _desision_tree()
    client.deploy('sub', sub, 'Simple subtraction function.', override=True)
    client.deploy('clustering',
                  clustering,
                  'Returns cluster Ids for each data point specified by the '
                  'pairs in x and y',
                  override=True)
    client.deploy('decision_tree',
                  decision_tree,
                  'decision_tree',
                  override=True)
