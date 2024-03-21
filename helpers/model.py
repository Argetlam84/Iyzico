import lightgbm as lgb
import feature_engineering as fe

def train_lightgbm_model(X_train, Y_train, X_val, Y_val, cols, lgb_params):
    lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
    lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

    # verbose_eval parametresini lgb_params içine ekleyelim
    lgb_params['verbose_eval'] = 100

    model = lgb.train(lgb_params, lgbtrain,
                      valid_sets=[lgbtrain, lgbval],
                      valid_names=['train', 'val'],
                      num_boost_round=lgb_params['num_boost_round'],
                      feval=fe.lgbm_smape)

    return model

def predict_with_lightgbm_model(model, X):
    return model.predict(X, num_iteration=model.best_iteration)