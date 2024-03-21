import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb


def plot_transaction_count(df, merchant_id, start_year, end_year):
    num_years = end_year - start_year
    fig, axes = plt.subplots(1, num_years, figsize=(15 * num_years, 5))

    for i, year in enumerate(range(start_year, end_year)):
        df_subset = df[(df.merchant_id == merchant_id) &
                       (df.transaction_date >= str(year) + "-01-01") &
                       (df.transaction_date < str(year + 1) + "-01-01")]
        ax = axes[i]
        df_subset["Total_Transaction"].plot(ax=ax)
        ax.set_title(str(merchant_id) + ' ' + str(year) + '-' + str(year + 1) + ' Transaction Count')
        ax.set_xlabel('')

    plt.tight_layout()
    plt.show()


def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


