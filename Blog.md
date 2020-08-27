**Understanding when to use k-Nearest Neighbor**
Imagine you wanted to predict an outcome knowing a given set of aspects. The prediction could be anything from accurately guessing the price of a house to finding the NBA player that is most similar to Michael Jordan. In order to make this prediction we use a set of known aspects. For our house example, these known variables include housing prices in a neighboring area, square footage, or if a house has tile flooring. To find the player that is like-Mike, maybe we’d examine scoring, shot location preference, and player position. Using these variables to examine what house or which player might have the stats that closest resemble what we are searching for.

Now, one particular algorithm that can be utilized for this is k-Nearest Neighbors, a machine learning algorithm useful for classification(identify the category/class to which a new data will fall under) and regression(targeting a dependent variable based on independent variables).

At a high level, the k-Nearest Neighbor(KNN) algorithm calculates the most similar data points given a specific set of inputs. The algorithm specifically uses a metric called “distance” which analyzes the spread between each point of a given dataset. So if a data set contain a list of houses with area codes, it would analyze each area code to determine the closest houses given the area code. Within the category of distance are more specific metrics including Euclidean distance and Jaccard distance, which are useful for numeric data points and boolean values, respectively. There are, of course other algorithms within the nearest neighbor family such as ball tree and kdtree.

To implement your own KNN model, you’ll first want to determine what problem you’re trying to solve and what KNN model makes the most sense for your specific problem/task.

The advantages of using KNN include:
* It’s rather easy to implement in a short amount of time
* There are several standard libraries that are well documented
* As we discussed earlier, KNN can be used for classification or regression problems
* No training is required before making predictions

Obviously, this comes with a set of counter-tradeoffs including:
* Since the algorithm compares every single point to one another, it gets slower as the data set grows larger or as the feature set(independent variables) increases
* The accuracy depends on the quality of the data
* The data will be sensitive to outliers or missing values

The following codebase in my repo utilizes KNN model on a former build week project that my team did analyze weed strains. The MedCab project used a KNN model (with a kd tree) to analyze strains and determine which strain a user would want given their desired inputs(happy, sad, hungry, etc). 

While you could create your own KNN model, the easier and more practical path is to use standard libraries that exist. If you’re looking to just test out KNN models, the iris data set is an easy example, however, I’d recommend finding some data you’re personally interested in. 

Once you’ve done some rudimentary cleaning and assigned binary to any non-numeric variables it should be somewhat easy to implement the KNN model from there. After you’ve done that I’d recommend trying to implement other models like decision trees to see how the accuracy compares to KNN. 
