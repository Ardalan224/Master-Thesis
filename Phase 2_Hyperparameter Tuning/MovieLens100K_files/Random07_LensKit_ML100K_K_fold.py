from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Random  # Import Random algorithm
from sklearn.model_selection import train_test_split
import pandas as pd
from lenskit.datasets import ML100K
import seedbank

if __name__ == '__main__':
    # Initialize seed
    seedbank.initialize(1)

    # Function to down-sample the dataset
    def downsample_data(ratings, percentage):
        sampled_ratings = ratings.sample(frac=percentage, random_state=1)
        return sampled_ratings


    # Load and preprocess the dataset
    file_path = '/home/g063231/Master_Thesis/Running/ml-100k'
    ml100k = ML100K(file_path)
    ratings = ml100k.ratings
    ratings = ratings.dropna(subset=['rating'])

    print(ratings.head())

    # Split into train and test sets
    train_data, test_data = train_test_split(ratings, test_size=0.1, random_state=1)

    # Down-sample the training data
    train_data = downsample_data(train_data, percentage=0.7)

    # Function to evaluate with nDCG metric
    def evaluate_with_ndcg(aname, algo, train, valid):
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train)
        users = valid.user.unique()
        recs = batch.recommend(fittable, users, 10)
        recs['Algorithm'] = aname
        return recs

    # Perform k-fold cross-validation and compute nDCG
    results = []
    algo_random = Random(rng_spec=1)  # Use Random algorithm
    all_recs = []
    valid_data = []

    fold_num = 1

    for train, valid in xf.partition_users(train_data[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2), rng_spec=1):
        valid_data.append(valid)
        fold_recs = evaluate_with_ndcg('Random', algo_random, train, valid)
        all_recs.append(fold_recs)

        # Compute nDCG for this fold
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg)
        fold_results = rla.compute(fold_recs, valid)
        mean_ndcg = fold_results['ndcg'].mean()

        print(f"Fold {fold_num}: Mean nDCG = {mean_ndcg:.4f}")
        fold_num += 1

    all_recs = pd.concat(all_recs, ignore_index=True)
    valid_data = pd.concat(valid_data, ignore_index=True)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    fold_results = rla.compute(all_recs, valid_data)

    # Calculate mean nDCG directly from fold_results
    mean_ndcg = fold_results['ndcg'].mean()
    results.append({'Algorithm': 'Random', 'Mean nDCG': mean_ndcg})

    # Print results
    print("Results:")
    for result in results:
        print(f"Algorithm = {result['Algorithm']}: Mean nDCG = {result['Mean nDCG']:.4f}")

    # Fit the algorithm on the full training data
    algo_random_full = Recommender.adapt(util.clone(algo_random))
    algo_random_full.fit(train_data)
    print(len(test_data['user']))
    # Check the number of unique users in the test set
    num_test_users = test_data['user'].nunique()
    print(f"Number of unique users in the test set: {num_test_users}")

    # Get the unique users from the test data
    test_users = test_data.user.unique()

    # Generate recommendations for the test users with batch processing
    test_recs = batch.recommend(algo_random_full, test_users, 10)
    test_recs['Algorithm'] = 'Random'

    # Evaluate the recommendations using the test set
    test_rla = topn.RecListAnalysis()
    test_rla.add_metric(topn.ndcg)
    test_results = test_rla.compute(test_recs, test_data)

    # Display the evaluation results
    print(test_results.head())
    print("NDCG mean for test set:", test_results.groupby('Algorithm').ndcg.mean())
