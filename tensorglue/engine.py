import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import svds
from collections import namedtuple
from tensorglue.lib.hosvd import tucker_als


class RecommenderData(object):
    _std_fields = ('userid', 'itemid', 'contextid', 'values')

    def __init__(self, data, userid, itemid, values=None, context_data=None, timeid=None):

        #add context data - invalid context records will be filtered out
        data, contextid = self.process_context(data, context_data, itemid)

        #aggregate duplicate records if any
        dup_fields = [userid, itemid, contextid] if contextid else [userid, itemid]
        data, values = self._aggregate_data(data, dup_fields, values, timeid)

        #initialize instance attributes
        self._data = data
        self.fields = namedtuple('Fields', self._std_fields)._make(map(eval, self._std_fields))
        self.arrange_by = timeid or contextid or values

        self.index = namedtuple('DataIndex', self._std_fields[:3])._make([None]*3)
        if contextid:
            self.index = self.index._replace(contextid=self.reindex(self._data, contextid))
        #_chunk used to split tensor decompositions into smaller pieces in memory
        self._chunk = 1000


    @staticmethod
    def process_context(data, context_data, join_col):
        if isinstance(context_data, str): #assumes there's corresponding column in data
            contextid = context_data
            data = data.copy()
        elif isinstance(context_data, pd.DataFrame): #assumes two-column input
            cid = 1 - context_data.columns.tolist().index(join_col)
            contextid = context_data.columns[cid]
            context_data = context_data.drop_duplicates(keep='last')
            data = pd.merge(data, context_data, on=join_col, how='inner', indicator=False)
            #TODO: notify about filtered items
        elif context_data is None:
            contextid = None
            data = data.copy()
        else:
            raise ValueError
        return data, contextid


    @staticmethod
    def aggregate(data, cols, by_values=None, timeid=None):
        agg_time = 'max'
        if by_values is None:
            by_values, agg_vals = 'counts', 'size'
            agg_cols = [agg_time, agg_vals]
            agg_rule = {timeid: agg_cols} if timeid else agg_vals
        else:
            agg_vals = 'sum'
            agg_rule = {timeid: agg_time, by_values: agg_vals} if timeid else agg_vals

        agg_data = data.groupby(cols, sort=False).agg(agg_rule)

        if isinstance(agg_data, pd.Series):
            agg_data = agg_data.to_frame(by_values)
        elif agg_data.columns.nlevels==2:
            agg_data = agg_data[timeid].rename(columns=dict(zip(agg_cols, [timeid, by_values])))

        return agg_data.reset_index(), by_values


    @staticmethod
    def reindex(data, col, sort=True, inplace=True):
        grouper = data.groupby(col, sort=sort).grouper
        new_idx = grouper.group_info[1]
        old_idx = grouper.levels[0]
        if inplace:
            data.loc[:, col] = grouper.group_info[0]
            result = pd.DataFrame({'old': old_idx, 'new': new_idx})
        else:
            result = grouper.group_info[0]
        return result


    def _aggregate_data(self, data, idx_fields, values, timeid):
        '''This method is for forward compatibility with subclasses, which
        require different aggregation scheme. Not touching `aggregate` method
        because it may be used not only during initialization => must be static.
        '''
        #pandas duplicated works in v. 17.1 or earlier than 17.0
        #there was a bug in v. 17.0
        has_duplicates = data.duplicated(idx_fields, keep='last').any()
        if has_duplicates:
            print 'Duplicates found. Aggregating data by {}.'.format(idx_fields)
            data, values = self.aggregate(data, idx_fields, values, timeid)
        return data, values


    def _prepare_data(self, test_ratio=0.2, eval_num=3):
        #TODO: make this a property, so that changing it would lead to data splitting
        self.test_ratio = test_ratio
        self.eval_num = eval_num

        self._split_test_data()
        self._reindex_data()
        self._align_test_items()
        self._split_eval_data()


    def _split_test_data(self):
        userid = self.fields.userid
        itemid = self.fields.itemid
        test_max_ratio = self.test_ratio
        eval_items_num = self.eval_num
        data = self._data

        test_users_max = (self._data[userid].nunique()) * test_max_ratio
        if eval_items_num:
            #sort=True makes splitting identical for data with or w/o context - good for debugging
            test_users_sel = data.groupby(userid, sort=True)[itemid].nunique() > eval_items_num
            test_users_all = test_users_sel.sum()
            if test_users_all < test_users_max:
                print 'Warning: there\'s not enough users satisfying test set criterias.'
            test_users_num = min(test_users_all, test_users_max)
        else:
            raise NotImplementedError
            #test_users_num = test_users_max

        #TODO: add option to choose users randomly
        test_users_pos = test_users_sel.cumsum() > (test_users_all - test_users_num)
        test_users_filter = test_users_sel & test_users_pos
        test_users_idx = test_users_filter.index[test_users_filter]

        train_users_filter = ~test_users_filter
        train_users_idx = train_users_filter.index[train_users_filter]

        self.train = self._data[self._data[userid].isin(train_users_idx)].copy()#.sort_values([userid, itemid])
        self.test = self._data[self._data[userid].isin(test_users_idx)].copy()#.sort_values([userid, itemid])


    def _reindex_data(self):
        userid, itemid = self.fields.userid, self.fields.itemid
        reindex = self.reindex
        user_index = [reindex(data, userid, sort=False) for data in [self.train, self.test]]
        user_index = namedtuple('UserIndex', 'train test')._make(user_index)
        self.index = self.index._replace(userid=user_index)
        self.index = self.index._replace(itemid=reindex(self.train, itemid))


    def _align_test_items(self):
        #TODO: add option to filter by whole sessions, not just items
        items_index = self.index.itemid.set_index('old')
        itemid = self.fields.itemid
        self.test.loc[:, itemid] = items_index.loc[self.test[itemid].values, 'new'].values
        # need to filter those items which were not in train set
        if self.test[itemid].isnull().any():
            print 'Unseen items found in the test set. Dropping...'
            userid = self.fields.userid
            self.test.dropna(axis=0, subset=[itemid], inplace=True)
            # there could be insufficient data now => filter again
            valid_users_sel = self.test.groupby(userid, sort=False)[itemid].nunique() > self.eval_num
            valid_users_idx = valid_users_sel.index[valid_users_sel]
            self.test = self.test.loc[self.test[userid].isin(valid_users_idx)]
            #reindex the test userids as they were filtered
            new_test_idx = self.reindex(self.test, userid, sort=False, inplace=True)
            #update index info accordingly
            old_test_idx = self.index.userid.test
            self.index.userid._replace(test=old_test_idx[old_test_idx['new'].isin(valid_users_idx)])
            self.index.userid.test.loc[new_test_idx['old'].values, 'new'] = new_test_idx['new'].values


    def _split_eval_data(self):
        userid, itemid, contextid, values = self.fields
        lastn = self.eval_num
        arrange_by = self.arrange_by

        if arrange_by == values:
            if contextid is None: #pure 2D
                #assert ~self.test.has_duplicates(subset=[userid, itemid]).any()
                #TODO: maybe instead of assertion use smth like:
                #self.test.pivot_table(index=[userid, itemid], values=values, aggfunc='max').to_frame(values).reset_index()
                eval_grouper = self.test.groupby(userid, sort=False)[values]
                eval_idx = eval_grouper.nlargest(lastn).index.get_level_values(1)
            else: #contextual 2D - this is for modified SVD
                print 'Summarizing contextual values'
                context_values = self._contextualize(self.test).to_frame(values)

                self.test.loc[:, values] = context_values[values]

                context_values['index'] = context_values.index
                #summarize context values, preserve indices for all contexts of an item
                flat_values = context_values\
                                .groupby([self.test[userid], self.test[itemid]], sort=False)\
                                    .agg({'index': lambda x:tuple(x), values: 'sum'})
                eval_grouper = flat_values.groupby(level=userid, sort=False)
                #aggregate indices for evaluation set
                eval_idx = eval_grouper.apply(lambda x: x.nlargest(lastn, values)['index']).sum()
                eval_idx = list(eval_idx)
        elif arrange_by == contextid: #full context, Tensor-based approach
            print 'Maximizing contextual values'
            context_values = self._contextualize(self.test).to_frame(values)

            self.test.loc[:, values] = context_values[values]

            context_values['index'] = context_values.index
            #maximize context values, preserve indices for all contexts of an item
            flat_values = context_values\
                            .groupby([self.test[userid], self.test[itemid]], sort=False)\
                                .agg({'index': lambda x:tuple(x), values: 'max'})
            eval_grouper = flat_values.groupby(level=userid, sort=False)
            #aggregate indices for evaluation set
            eval_idx = eval_grouper.apply(lambda x: x.nlargest(lastn, values)['index']).sum()
            eval_idx = list(eval_idx)
        else: #timeid (if smth else, pandas will throw KeyError anyway)
            if contextid:
                context_values = self._contextualize(self.test)
                self.test.loc[:, values] = context_values

            timeid = arrange_by
            time_data = self.test[[timeid]].copy()
            time_data['index'] = time_data.index
            flat_values = time_data\
                            .groupby([self.test[userid], self.test[itemid]], sort=False)\
                                .agg({'index': lambda x:tuple(x), timeid: 'max'})
            eval_grouper = flat_values.groupby(level=userid, sort=False)
            eval_idx = eval_grouper.apply(lambda x: x.nlargest(lastn, timeid)['index']).sum()
            eval_idx = list(eval_idx)

        evalset = self.test.loc[eval_idx]
        #TODO: add option to sample small piece of testset randomly
        testset = self.test[~self.test.index.isin(eval_idx)]
        self.test = namedtuple('TestData', 'testset evalset')._make([testset, evalset])


    def _contextualize(self, data, normalize=False):
        userid, itemid, contextid, values = self.fields
        #assumes the data was aggregated - e.g. no duplicate triplets {userid, itemid, contextid}:
        num_contexts = data.groupby([userid, itemid], sort=False)[contextid].transform('size')
        weighted_values = data[values] / num_contexts

        num_items = data.groupby(userid, sort=False)[itemid].transform('nunique') #could be `size` if no context
        normed_values = data.groupby([userid, contextid], sort=False)[values].transform('sum') / num_items

        contextual_values = (weighted_values * normed_values)
        if normalize:
            #TODO: make max_val_weighted and max_val_normed, e.g. divide by a factor specific to each user
            #max_val_normed = normed_values.groupby(data[userid], sort=False).transform('max')
            #max_val_weighted = weighted_values.groupby(data[userid], sort=False).transform('max')
            #norm_coef = (max_val_weighted * max_val_normed)
            norm_coef = contextual_values.groupby(data[userid], sort=False).transform('max')
            contextual_values /= norm_coef

        return contextual_values


    def _to_coo(self):
        userid, itemid, contextid, values = self.fields
        if contextid:
            val = self._contextualize(self.train).values
            idx_fields = [userid, itemid, contextid]
        else:
            val = self.train[values].values
            idx_fields = [userid, itemid]

        idx = self.train[idx_fields].values
        shp = self.train[idx_fields].max() + 1

        idx = np.ascontiguousarray(idx)
        val = np.ascontiguousarray(val)
        return idx, val, shp


    def train_model(self, model, svd_rank=10, tensor_ranks=(13, 8, 12)):
        userid, itemid, contextid, values = self.fields
        if model.lower() == 'svd':
            self._get_recommendations = self.svd_recommender
            svd_idx = (self.train[userid].values,
                       self.train[itemid].values)
            #TODO: needs to be more failproof with self.arrange_by and contextid
            if contextid:
                svd_val = self._contextualize(self.train).values
            else:
                svd_val = self.train[values].values
            #the data is reindexed - no need to specify shape
            #svd_shp = self.train[[userid, itemid]].max()+1
            #.tocsr() will accumulate duplicates values (having different context)
            svd_matrix = sp.sparse.coo_matrix((svd_val, svd_idx),
                                              dtype=np.float64).tocsr() #shape=svd_shp

            _, _, items_factors = svds(svd_matrix, k=svd_rank, return_singular_vectors='vh')
            self._items_factors = np.ascontiguousarray(items_factors[::-1, :])

        elif model.lower() == 'i2i':
            if contextid:
                raise NotImplementedError

            self._get_recommendations = self.i2i_recommender
            i2i_matrix = self._build_i2i_matrix()
            self._i2i_matrix = i2i_matrix

        elif model.lower() == 'tensor':
            self._get_recommendations = self.tensor_recommender
            idx, val, shp = self._to_coo()
            _, items_factors, context_factors, _ = tucker_als(idx, val, shp, tensor_ranks, growth_tol=0.001)
            self._items_factors = items_factors
            self._context_factors = context_factors

        else:
            raise NotImplementedError


    def evaluate(self, topk=10):
        all_recs = self._get_recommendations()
        top_recs = np.argpartition(all_recs, -topk, axis=1)[:, -topk:]
        scores = self._get_scores(top_recs)
        return scores


    def svd_recommender(self):
        userid, itemid, contextid, values = self.fields
        test_idx = (self.test.testset[userid].values,
                    self.test.testset[itemid].values)
        if contextid:
            #TODO: refactor it! need to think about dependence on self.arrange_by and contextid
            #values are contextualized already
            test_val = self.test.testset[values].values
        else:
            test_val = self.test.testset[values].values

        v = self._items_factors
        test_shp = (self.test.testset[userid].max()+1,
                    v.shape[1])

        test_matrix = sp.sparse.coo_matrix((test_val, test_idx),
                                           shape=test_shp,
                                           dtype=np.float64).tocsr()

        svd_scores = (test_matrix.dot(v.T)).dot(v)
        return svd_scores


    def tensor_recommender(self):
        userid, itemid, contextid, values = self.fields
        v = self._items_factors
        w = self._context_factors

        #TODO: split calculation into batches of users so that it doesn't
        #blow computer memory out.
        test_shp = (self.test.testset[userid].max()+1, v.shape[0], w.shape[0])
        idx_data = self.test.testset.loc[:, [userid, itemid, contextid]].values.T.astype(np.int64)
        idx_flat = np.ravel_multi_index(idx_data, test_shp)
        shp_flat = (test_shp[0]*test_shp[1], test_shp[2])
        idx = np.unravel_index(idx_flat, shp_flat)
        #values are assumed to be contextualized already
        val = self.test.testset[values].values
        test_tensor_mat = sp.sparse.coo_matrix((val, idx), shape=shp_flat).tocsr()

        tensor_scores = np.empty((test_shp[0], test_shp[1]))
        chunk = self._chunk
        for i in xrange(0, test_shp[0], chunk):
            start = i
            stop = min(i+chunk, test_shp[0])

            test_slice = test_tensor_mat[start*test_shp[1]:stop*test_shp[1], :]
            slice_scores = test_slice.dot(w).reshape(stop-start, test_shp[1], w.shape[1])
            slice_scores = np.tensordot(slice_scores, v, axes=(1, 0))
            slice_scores = np.tensordot(np.tensordot(slice_scores, v, axes=(2, 1)), w, axes=(1, 1))
            tensor_scores[start:stop, :] = slice_scores.max(axis=2)

        return tensor_scores


    def _get_scores(self, recs):
        userid, itemid = self.fields.userid, self.fields.itemid
        #no need for context here -> drop duplicates
        #but it's unstable in pandas v. 17.0, only works in 17.1 or <17.0
        eval_data = self.test.evalset.drop_duplicates(subset=[userid, itemid]).sort_values(userid)[itemid]
        #TODO: sort only if not monotonic
        eval_matrix = eval_data.values.reshape(-1, self.eval_num).astype(np.int64)
        scores = (recs[:, :, None] == eval_matrix[:, None, :]).sum()
        return scores


    def _build_i2i_matrix(self):
        userid, itemid = self.fields.userid, self.fields.itemid

        shape = self.train[[userid, itemid]].max() + 1
        user_item = sp.sparse.coo_matrix((np.ones_like(self.train[userid].values),
                                      (self.train[userid].values, self.train[itemid].values)),
                                      shape=shape).tocsc()
                                      
        i2i_matrix = user_item.T.dot(user_item)
        #exclude self-links
        diag_vals = i2i_matrix.diagonal()
        i2i_matrix -= sp.sparse.dia_matrix((diag_vals, 0), shape=i2i_matrix.shape)
        #see http://nbviewer.jupyter.org/gist/Midnighter/9992103 for benchmark
        return i2i_matrix


    def i2i_recommender(self):
        userid, itemid = self.fields.userid, self.fields.itemid
        test_idx = (self.test.testset[userid].values,
                    self.test.testset[itemid].values)
        test_val = np.ones(self.test.testset.shape[0],)
        test_shp = (self.test.testset[userid].max()+1,
                    self.index.itemid.new.max() + 1)

        test_matrix = sp.sparse.coo_matrix((test_val, test_idx),
                                           shape=test_shp,
                                           dtype=np.float64).tocsr()
        #exploiting simmetric property of i2i matrix here
        i2i_scores = test_matrix.dot(self._i2i_matrix)
        return i2i_scores.A


class RecommenderDataTyped(RecommenderData):
    '''Subclass takes additional argument for initialization - type_aggregation.
    This is used to take into account transaction type information in retail data.
    '''
    def __init__(self, *args, **kwargs):

        type_aggregation = kwargs.pop('type_aggregation', None)
        if isinstance(type_aggregation, dict):
            [(typeid, type_weights)] = type_aggregation.iteritems()
            self.typeid = typeid
        elif isinstance(type_aggregation, str):
            self.typeid = type_aggregation
            type_weights = None
        else:
            self.typeid = type_weights = None

        super(RecommenderDataTyped, self).__init__(*args, **kwargs)

        if type_weights:
            self.types = self._data[typeid].unique()
            for i, val in enumerate(self.types):
                self._data.loc[self._data[typeid]==val, self.fields.values] *= type_weights[i]


    def _aggregate_data(self, data, idx_fields, values, timeid):
        if self.typeid:
            #don't aggregate data by type - need it later
            idx_fields.append(self.typeid)
        return super(RecommenderDataTyped, self)._aggregate_data(data, idx_fields, values, timeid)


    def _prepare_data(self, test_ratio=0.2, eval_num=3):
        self.test_ratio = test_ratio
        self.eval_num = eval_num

        self._split_test_data()

        if self.typeid:
            zero_entries = self.train[self.fields.values]==0
            #don't feed zeroes into sparse train matrix
            if zero_entries.any():
                    self.train = self.train[~zero_entries].copy()

        self._reindex_data()
        self._align_test_items()
        self._split_eval_data()


    def _split_eval_data(self):
        typeid = self.typeid
        if typeid:
            userid, itemid, contextid, values = self.fields
            timeid = None if self.arrange_by in [values, contextid] else self.arrange_by
            agg_cols = [userid, itemid, contextid] if contextid else [userid, itemid]

            if timeid:
                super(RecommenderDataTyped, self)._split_eval_data()
                testset, _ = self.aggregate(self.test.testset, agg_cols, by_values=values, timeid=timeid)
                #TODO: may not always be correct - not always max typeid corresponds to max timeid
                evalset = self.test.evalset.groupby([userid, itemid], sort=False, as_index=False)\
                                            .agg({typeid:'max', timeid:'max'})
                self.test = namedtuple('TestData', 'testset evalset')._make([testset, evalset])
            else:
                self.test, _ = self.aggregate(self.test, agg_cols, by_values=values, timeid=timeid)
                #TODO: if contextid is not None and not all type_weights > 0:
                #have to remove zero-valued entries BEFORE contextualization procedure!!!
                super(RecommenderDataTyped, self)._split_eval_data()

            zero_entries = self.test.testset[self.fields.values]==0
            if zero_entries.any():
                #don't feed zeroes into sparse test matrix
                self.test._replace(testset=self.test.testset[~zero_entries].copy())
        else:
            super(RecommenderDataTyped, self)._split_eval_data()


    def train_model(self, *args, **kwargs):
        if self.typeid:
            userid, itemid, contextid, values = self.fields
            timeid = None if self.arrange_by in [values, contextid] else self.arrange_by
            agg_cols = [userid, itemid, contextid] if contextid else [userid, itemid]
            self.train, _ = self.aggregate(self.train, agg_cols, by_values=values, timeid=timeid)

        super(RecommenderDataTyped, self).train_model(*args, **kwargs)


    def _get_scores(self, recs):
        userid, itemid = self.fields.userid, self.fields.itemid
        typeid = self.typeid
        if typeid:
            #no need for context here -> drop duplicates, works in pandas v. 17.1
            eval_data = self.test.evalset.drop_duplicates(subset=[userid, itemid]).sort_values(userid)[[itemid, typeid]]
            #eval_data = self.test.evalset.groupby([userid, itemid], sort=False, as_index=False)\
            #                             .first().sort_values(userid)[[itemid, typeid]]
            eval_matrix = eval_data.set_index(typeid, append=True).unstack()[itemid]\
                                  .fillna(-1).astype(np.int64).values.reshape(-1, self.eval_num, self.types.size)
            scores = (recs == eval_matrix.transpose(2, 1, 0)[..., None]).sum(axis=(1, 2, 3))
            #scores =  namedtuple('Scores', ['type{}'.format(i) for i in map(str, self.types)])._make(scores_by_type)
        else:
            scores = super(RecommenderDataTyped, self)._get_scores(recs)
        return scores
