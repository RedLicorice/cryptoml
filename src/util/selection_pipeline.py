from sklearn.pipeline import Pipeline as SklearnPipeline

# This class exposes coef_ and feature_importances_ from a pipeline in order to
#  use it for feature selection in wrapper methods such as RFECV or SelectFromModel
class Pipeline(SklearnPipeline):
    train_x = None
    train_y = None
    is_fit = False

    def fit(self, X, y=None, **fit_params):
        # Keep record of training datasets
        self.train_x = X,
        self.train_y = y

        super(Pipeline, self).fit(X, y, **fit_params)
        # We're assuming classifier is the last element of the pipeline
        clf = self.steps[-1][-1]
        if hasattr(clf, 'coef_'):
            self.coef_ = clf.coef_
        if hasattr(clf, 'feature_importances_'):
            self.feature_importances_ = clf.feature_importances_

        self.is_fit = True
        return self