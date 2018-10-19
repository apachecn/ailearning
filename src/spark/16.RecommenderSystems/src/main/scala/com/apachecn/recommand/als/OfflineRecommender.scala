package apache.wiki

import scala.collection.Map

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD

/**
 * @author ${user.name}
 * Copyright 2015 Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 * 
 * 参考地址
 * GitHub: https://github.com/apachecn/RecommenderSystems
 * 推荐系统: http://www.kuqin.com/shuoit/20151202/349305.html
 * ALS说明: http://www.csdn.net/article/2015-05-07/2824641
 */

// UserMovies对象： 用户名，电影ID，评分
case class MovieRating(userID: String, movieID: Int, rating: Double) extends scala.Serializable

object OfflineRecommender {

    // 推荐函数
    def recommend(sc: SparkContext, rawUserMoviesData: RDD[String], rawHotMoviesData: RDD[String], base: String): Unit = {
       
        // 抽取 电影ID和电影名称的映射关系
        // 设置广播变量
        val moviesAndName = buildMovies(rawHotMoviesData)
        val bMoviesAndName = sc.broadcast(moviesAndName)

        // 解析本文数据，并进行格式化转换处理
        val data = buildRatings(rawUserMoviesData)

        // 用户名称+索引ID[Long]
        val userIdToInt: RDD[(String, Long)] = data.map(_.userID)
            .distinct()
            .zipWithUniqueId()
        // 索引ID+用户名称
        val reverseUserIDMapping: RDD[(Long, String)] = userIdToInt map { case (l, r) => (r, l) }
        val bReverseUserIDMap = sc.broadcast(reverseUserIDMapping.collectAsMap())
        // 用户名称+索引ID[Int]
        val userIDMap: Map[String, Int] = userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }
        val bUserIDMap = sc.broadcast(userIDMap)

        // 转换数据类型为 RDD[Rating] 类型
        val ratings: RDD[Rating] = data.map { 
            r => Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
        }.cache()

        //使用协同过滤算法建模
        /* 
         * ALS是alternating least squares的缩写 , 意为交替最小二乘法。
         * 该方法常用于基于矩阵分解的推荐系统中。例如：将用户(user)对商品(item)的评分矩阵分解为两个矩阵：一个是用户对商品隐含特征的偏好矩阵，另一个是商品所包含的隐含特征的矩阵。
         * 在这个矩阵分解的过程中，评分缺失项得到了填充，也就是说我们可以基于这个填充的评分来给用户最商品推荐了。
         * 由于评分数据中有大量的缺失项，传统的矩阵分解SVD（奇异值分解）不方便处理这个问题，而ALS能够很好的解决这个问题。
         * 
         * 
         * 训练模型接下来调用ALS.train()方法，进行模型训练： 【ALS.train()显式反馈; ALS.trainImplicit() 隐式反馈】
         * 显式评分时，每个用户对于一个产品的评分需要是一个得分(例如1到5星)
         * 隐式反馈时，每个评分代表的是用户会和给定产品发送交互的置信度(比如随着用户访问一个网页次数的增加，平跟也会提高)，预测出来的也是置信度。
         * 
         * 
         * 参数说明：
         * ratings: RDD[Rating]  
         * rank  要使用的features的数量, 模型中隐藏因子的个数
         * iterations 迭代的次数，推荐值：10-20
         * lambda  正则化参数【惩罚函数的因数，是ALS的正则化参数，推荐值：0.01】
         * alpha 用来在ALS中计算置信度的常量，默认1.0 【隐式反馈】
         */

        //val model = ALS.trainImplicit(ratings, 10, 10, 0.01, 1.0)
        val model = ALS.train(ratings, 50, 10, 0.0001)
        // unpersist用于删除磁盘、内存中的相关序列化对象
        ratings.unpersist()

        model.save(sc, base + "output/model")
        //val sameModel = MatrixFactorizationModel.load(sc, base + "model")

        // 每个用户返回5部电影，格式：(user, ratings)
        val allRecommendations = model.recommendProductsForUsers(5) map {
            case (userID, recommendations) => {
                var recommendationStr = ""
                for (r <- recommendations) {
                    recommendationStr += r.product + ":" + bMoviesAndName.value.getOrElse(r.product, "") + ","
                }
                if (recommendationStr.endsWith(","))
                    recommendationStr = recommendationStr.substring(0,recommendationStr.length-1)

                (bReverseUserIDMap.value.get(userID).get,recommendationStr)
            }
        }

        // allRecommendations.saveAsTextFile(base + "output/result.csv")
        allRecommendations.coalesce(1, true).sortByKey().saveAsTextFile(base + "output/result.csv")

        // 自定义方法，删除model内部的序列化对象
        unpersist(model)
    }


    def main(args: Array[String]): Unit = {
        // 初始化 SparkContext
        val conf = new SparkConf().setMaster("local").setAppName("OfflineRecommender")
        val sc = new SparkContext(conf)
        val base = if (args.length > 0) args(0) else "file:/opt/git/RecommenderSystems/"

        // 导入数据，获取RDD
        // UserMovies格式： 用户名，电影ID，评分
        // HotMovies 格式： 电影ID，评分，电影名称
        val rawUserMoviesData = sc.textFile(base + "input/user_movies.csv")
        val rawHotMoviesData  = sc.textFile(base + "input/hot_movies.csv")

        // 抽样检查数据
        preparation(rawUserMoviesData, rawHotMoviesData)
        println("准备完数据")
        // 抽样评估推荐结果
        // model(sc, rawUserMoviesData, rawHotMoviesData)
        // 整体推荐评分的评估
        // evaluate(sc,rawUserMoviesData, rawHotMoviesData)

        recommend(sc, rawUserMoviesData, rawHotMoviesData, base)

        // 每一个 JVM 可能只能激活一个 SparkContext 对象。在创新一个新的对象之前，必须调用 stop() 该方法停止活跃的 SparkContext。
        sc.stop()
    }


    // 检查数据
    def preparation(rawUserMoviesData: RDD[String], rawHotMoviesData: RDD[String]) = {
        // 格式：（用户名，索引分区ID[分区整除／分区余1/。。／分区余n]） => 计算结果：索引分区ID的统计信息[("最大值:" + stats.max, "最小值:" + stats.min, "平均值:" + stats.mean, "方差" + stats.variance, "标准方差" + stats.stdev, "元素个数:" + stats.count)]
        // zipWithUniqueId 可参考： http://lxw1234.com/archives/2015/07/352.htm
        val userIDStats = rawUserMoviesData.map(_.split(",")(0).trim)
            .distinct()
            .zipWithUniqueId()
            .map(_._2.toDouble)
            .stats()

        // 格式： 电影ID的集合，统计信息
        val itemIDStats = rawUserMoviesData.map(_.split(",")(1).trim.toDouble)
            .distinct()
            .stats()

        println(userIDStats)
        println(itemIDStats)

        val moviesAndName = buildMovies(rawHotMoviesData)

        // 获取第一行，head()=first(), head(n)=take(n)
        val (movieID, movieName) = moviesAndName.head
        println(movieID + " -> " + movieName)
    }


    // 获取电影ID和电影名字的RDD
    // collectAsMap: 在相同ID下，后面出现的Value会覆盖先出现的Value
    def buildMovies(rawHotMoviesData: RDD[String]): Map[Int, String] = rawHotMoviesData.flatMap { line =>
        val tokens = line.split(",")
        if (tokens(0).isEmpty) {
            None
        } else {
            Some((tokens(0).toInt, tokens(2)))
        }
    }.collectAsMap()


    // MovieRating电影评估字段分解和类型转化，格式： 用户名，电影ID，评分
    def buildRatings(rawUserMoviesData: RDD[String]): RDD[MovieRating] = {
        rawUserMoviesData.map { line =>
            val Array(userID, moviesID, countStr) = line.split(",").map(_.trim)
            var count = countStr.toInt
            count = if (count == -1) 3 else count
            MovieRating(userID, moviesID.toInt, count)
        }
    }

    
    // 自定义方法，删除model内部的序列化对象
    def unpersist(model: MatrixFactorizationModel): Unit = {
        // At the moment, it's necessary to manually unpersist the RDDs inside the model
        // when done with it in order to make sure they are promptly uncached
        model.userFeatures.unpersist()
        model.productFeatures.unpersist()
    }


    // http://stackoverflow.com/questions/27772769/spark-how-to-use-mllib-recommendation-if-the-user-ids-are-string-instead-of-co
    def model(sc: SparkContext, rawUserMoviesData: RDD[String], rawHotMoviesData: RDD[String]): Unit = {
        val moviesAndName = buildMovies(rawHotMoviesData)
        val bMoviesAndName = sc.broadcast(moviesAndName)
        val data = buildRatings(rawUserMoviesData)
        val userIdToInt: RDD[(String, Long)] = data.map(_.userID).distinct().zipWithUniqueId()
        val reverseUserIDMapping: RDD[(Long, String)] = userIdToInt map { case (l, r) => (r, l) }
        val userIDMap: Map[String, Int] = userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }
        val bUserIDMap = sc.broadcast(userIDMap)
        val ratings: RDD[Rating] = data.map { 
            r =>
            Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
        }.cache()

        //使用协同过滤算法建模
        //val model = ALS.trainImplicit(ratings, 10, 10, 0.01, 1.0)
        val model = ALS.train(ratings, 50, 10, 0.0001)
        ratings.unpersist()
        println("打印第一个userFeature")
        println(model.userFeatures.mapValues(_.mkString(", ")).first())

        for (userID <- Array(100,1001,10001,100001,110000)) {
            checkRecommenderResult(userID, rawUserMoviesData, bMoviesAndName, reverseUserIDMapping, model)
        }

        unpersist(model)
    }


    // 查看给某个用户的推荐
    def checkRecommenderResult(userID: Int, rawUserMoviesData: RDD[String], bMoviesAndName: Broadcast[Map[Int, String]], reverseUserIDMapping: RDD[(Long, String)], model: MatrixFactorizationModel): Unit = {

        val userName = reverseUserIDMapping.lookup(userID).head
        val recommendations = model.recommendProducts(userID, 5)
        //给此用户的推荐的电影ID集合
        val recommendedMovieIDs = recommendations.map(_.product).toSet

        //得到用户点播的电影ID集合
        val rawMoviesForUser = rawUserMoviesData.map(_.split(","))
            .filter { case Array(user, _, _) => user.trim == userName }
        val existingUserMovieIDs = rawMoviesForUser.map { 
                case Array(_, movieID, _) => movieID.toInt 
            }.collect().toSet


        println("用户" + userName + "点播过的电影名")
        //点播的电影名
        bMoviesAndName.value.filter { 
            case (id, name) => existingUserMovieIDs.contains(id) 
        }.values.foreach(println)

        println("推荐给用户" + userName + "的电影名")
        //推荐的电影名
        bMoviesAndName.value.filter { 
            case (id, name) => recommendedMovieIDs.contains(id) 
        }.values.foreach(println)
    }


    // 计算评估
    def evaluate( sc: SparkContext, rawUserMoviesData: RDD[String], rawHotMoviesData: RDD[String]): Unit = {
        val moviesAndName = buildMovies(rawHotMoviesData)
        // 解析用户电影的原数据
        val data = buildRatings(rawUserMoviesData)

        // 得到: (用户名, 索引ID) 和 (用户名: 索引ID)
        val userIdToInt: RDD[(String, Long)] = data.map(_.userID).distinct().zipWithUniqueId()
        val userIDMap: Map[String, Int] = userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }
        val bUserIDMap = sc.broadcast(userIDMap)

        val ratings: RDD[Rating] = data.map { 
            r => Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
        }.cache()

        val numIterations = 10


        // ALS.train 显式调用
        for (rank <- Array(10, 50); lambda <- Array(1.0, 0.01,0.0001)) {

            // 你可以对评分数据生成训练集和测试集，例如：训练集和测试集比例为8比2
            // val splits = ratings.randomSplit(Array(0.8, 0.2), seed = 111l)
            // val training = splits(0).repartition(numPartitions)
            // val test = splits(1).repartition(numPartitions)

            // 这里，我们是将评分数据全部当做训练集，并且也为测试集
            val model = ALS.train(ratings, rank, numIterations, lambda)

            // Evaluate the model on rating data
            // 评测我们要对比一下预测的结果，注意：我们将训练集当作测试集来进行对比测试。从训练集中获取用户和商品的映射：
            val usersMovies = ratings.map { 
                case Rating(user, movie, rate) => (user, movie)
            }
            println("实际评估的数量" + usersMovies.count)

            // 测试集的记录数等于评分总记录数，验证一下：
            val predictions = model.predict(usersMovies).map { 
                case Rating(user, movie, rate) => ((user, movie), rate)
            }
            println("预测评估的数量" + predictions.count)

            // 将真实评分数据集与预测评分数据集进行合并，这样得到用户对每一个商品的实际评分和预测评分：
            val ratesAndPreds = ratings.map { 
                case Rating(user, movie, rate) => ((user, movie), rate)
            }.join(predictions)

            /* 计算均方差：
             * 当然，我们不能凭着自己的感觉评价模型的好坏，尽管我们直觉告诉我们，这个结果看不错。我们需要量化的指标来评价模型的优劣。
             * 
             * 通过计算均方差（Mean Squared Error, MSE）来衡量模型的好坏。
             * 数理统计中均方误差是指参数估计值与参数真值之差平方的期望值，记为MSE。
             * MSE是衡量“平均误差”的一种较方便的方法，MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。
             * 
             * 我们可以调整rank，numIterations，lambda，alpha这些参数，不断优化结果，使均方差变小。
             * 比如：iterations越多，lambda较小，均方差会较小，推荐结果较优。
             */
            val MSE = ratesAndPreds.map { 
                case ((user, movie), (r1, r2)) => 
                val err = (r1 - r2)
                err * err
            }.mean()
            println(s"(rank:$rank, lambda: $lambda, Explicit ) Mean Squared Error = " + MSE)
        }

        // ALS.trainImplicit 隐式调用
        for (rank <- Array(10, 50); lambda <- Array(1.0, 0.01,0.0001); alpha <- Array(1.0, 40.0)) {
            
            //（trainImplicit 隐式调用／ train 显式调用）
            // 隐性的反馈（例如游览，点击，购买，喜欢，分享等等）
            // 
            // lambda 正则化参数;
            // alpha 用来在ALS中计算置信度的常量，默认1.0
            val model = ALS.trainImplicit(ratings, rank, numIterations, lambda, alpha)

            // Evaluate the model on rating data
            val usersMovies = ratings.map { 
                case Rating(user, movie, rate) => (user, movie)
            }
            val predictions = model.predict(usersMovies).map { 
                    case Rating(user, movie, rate) => ((user, movie), rate)
                }
            val ratesAndPreds = ratings.map { 
                case Rating(user, movie, rate) => ((user, movie), rate)
            }.join(predictions)

            val MSE = ratesAndPreds.map { 
                case ((user, movie), (r1, r2)) =>
                val err = (r1 - r2)
                err * err
            }.mean()
            println(s"(rank:$rank, lambda: $lambda,alpha:$alpha ,implicit    ) Mean Squared Error = " + MSE)
        }
    }
}