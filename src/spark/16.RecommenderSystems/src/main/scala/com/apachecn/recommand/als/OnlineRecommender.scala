// package apache.wiki

// import java.io.File
// import java.nio.charset.Charset

// import com.google.common.io.Files

// import org.apache.spark.{SparkConf, SparkContext}
// import org.apache.spark.broadcast.Broadcast
// import org.apache.spark.rdd.RDD
// import org.apache.spark.streaming.{Seconds, StreamingContext, Time}
// import org.apache.spark.util.{IntParam, LongAccumulator}
// /**
//  * @author ${user.name}
//  * Copyright 2015 Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
//  *
//  * See LICENSE file for further information.
//  * 
//  * 参考地址
//  * GitHub: https://github.com/apachecn/RecommenderSystems
//  * https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/streaming/RecoverableNetworkWordCount.scala
//  */

// object WordBlacklist {

//   @volatile private var instance: Broadcast[Seq[String]] = null

//   def getInstance(sc: SparkContext): Broadcast[Seq[String]] = {
//     if (instance == null) {
//       synchronized {
//         if (instance == null) {
//           val wordBlacklist = Seq("a", "b", "c")
//           instance = sc.broadcast(wordBlacklist)
//         }
//       }
//     }
//     return instance
//   }
// }

// /**
//  * Use this singleton to get or register an Accumulator.
//  */
// object DroppedWordsCounter {

//   @volatile private var instance: LongAccumulator = null

//   def getInstance(sc: SparkContext): LongAccumulator = {
//     if (instance == null) {
//       synchronized {
//         if (instance == null) {
//           instance = sc.longAccumulator("WordsInBlacklistCounter")
//         }
//       }
//     }
//     return instance
//   }
// }

// object OnlineRecommender{
//   def createContext(ip: String, port: Int, outputPath: String, checkpointDirectory: String): StreamingContext = {

//     // 如果已存在CheckPoint，就不进入该方法
//     println("Creating new context")

//     // Streaming处理的结果存放位置
//     val outputFile = new File(outputPath.split(":")(1))
//     if (outputFile.exists()) outputFile.delete()

//     val conf = new SparkConf().setAppName("OnlineRecommender")
//     // 默认本地模式运行
//     val isDebug = true
//     if (isDebug) {
//       conf.setMaster("local[2]")
//     }
//     // Create the context with a 1 second batch size
//     val ssc = new StreamingContext(conf, Seconds(10))
//     // checkpoint存放位置
//     ssc.checkpoint(checkpointDirectory)

//     // 创建一个将要连接到 hostname:port 的离散流，如 localhost:9999 
//     val lines = ssc.socketTextStream(ip, port)
//     // 将每一行拆分成单词 val words = lines.flatMap(_.split(" "))
//     val words = lines.flatMap(_.split(" "))
//     val wordCounts = words.map((_, 1)).reduceByKey(_ + _)
    
//     wordCounts.foreachRDD { (rdd: RDD[(String, Int)], time: Time) =>
//       // Get or register the blacklist Broadcast
//       val blacklist = WordBlacklist.getInstance(rdd.sparkContext)
//       // Get or register the droppedWordsCounter Accumulator
//       val droppedWordsCounter = DroppedWordsCounter.getInstance(rdd.sparkContext)
//       // Use blacklist to drop words and use droppedWordsCounter to count them
//       /*
//        * 累加器进行累加操作，blacklist.value的出现总次数
//        */
//       val counts = rdd.filter { case (word, count) =>
//         printf("blacklist.value=%s, word=%s, count=%d\n",  blacklist.value, word, count)
//         if (blacklist.value.contains(word)) {
//           droppedWordsCounter.add(count)
//           println("return false")
//           false
//         } else {
//           println("return true")
//           true
//         }
//       }.collect().mkString("[", ", ", "]")
//       val output = "Counts at time " + time + " " + counts
//       println(output)
//       println("Dropped " + droppedWordsCounter.value + " word(s) totally")
//       println("Appending to " + outputFile.getAbsolutePath)
//       Files.append(output + "\n", outputFile, Charset.defaultCharset())
//     }
//     return ssc
//   }


//   def main(args: Array[String]): Unit = {
    
//     val base = if (args.length > 0) args(0) else "file:/opt/git/RecommenderSystems/"

//     // 设置CheckPoint
//     val (ip, port, outputPath, checkpointDir) = ("localhost", 9999, base + "output/out", base + "output/checkpointDir")
//     val ssc = StreamingContext.getOrCreate(checkpointDir, () => createContext(ip, port, outputPath, checkpointDir))

//     ssc.start() // 启动计算 
//     ssc.awaitTermination() // 等待计算的终止
//   }
// }