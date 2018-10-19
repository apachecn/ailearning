package com.apachecn.recommand.colfliter

import breeze.linalg.SparseVector
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, SparseMatrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.collection.BitSet

/**
  * Created by ych on 2018/9/20.
  * 基于物品的协同过滤
  */
class ItemCF {

  def computeJaccardSimWithDF(sc: SparkContext,
                                  featurePath: String
                             ): CoordinateMatrix ={
    val sqlContext = new SQLContext(sc)
    val rdd = sqlContext.read.parquet(featurePath).select("features")
      .rdd.map(x=>x(0).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector])
      .map(x=>x.indices)
      .zipWithIndex()
      .map(x=>{
        for (i <- x._1) yield {
          (x._2, i)
        }
      })
      .flatMap(x=>x)
      .map(x=>(x._1,BitSet(x._2.toString.toInt)))
      .reduceByKey(_.union(_))

    val entries = rdd.cartesian(rdd).map {
      case ((key0, set0), (key1, set1)) => {
        val j = (set0 & (set1)).size
        val q = set0.union(set1).size
        val re = j.toDouble / q
        MatrixEntry(key0.toInt, key1.toInt, re)
      }
    }
    val simMat: CoordinateMatrix = new CoordinateMatrix(entries)
    simMat
  }


  def computeJaccardSimWithLibSVM(sc: SparkContext,
                        featurePath: String): CoordinateMatrix ={
    val rdd = sc.textFile(featurePath)
      .map(_.split(" ", 2)(1))
      .zipWithIndex()
      .map(x => (x._2.toInt,x._1.split(" ", -1)))
      .map(x=>{
        for (i <- x._2) yield {
          (i.split("\\:")(0), x._1)
        }
      }).flatMap(x=>x)
      .map(x=>(x._1,BitSet(x._2.toString.toInt)))
      .reduceByKey(_.union(_))

    val entries = rdd.cartesian(rdd).map {
      case((key0,set0),(key1,set1))=>{
        val j = (set0 &(set1)).size
        val q = set0.union(set1).size
        val re = j.toDouble/q
        MatrixEntry(key0.toInt,key1.toInt,re)
      }
    }
    val simMat: CoordinateMatrix = new CoordinateMatrix(entries)
    simMat
  }


  def computeCosSimWithDF(sc: SparkContext,
                              featurePath: String):  CoordinateMatrix ={

    val sqlContext = new SQLContext(sc)
    val rdd = sqlContext.read.parquet(featurePath).select("features")
      .rdd.map(x=>x(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector])
    val mat = new RowMatrix(rdd)
    val simMat = mat.columnSimilarities()
    simMat
  }

  /**
    *
    * @param sc
    * @param featureSize 特征数量
    * @param featurePath
    */
  def computeCosSimWithLibSVM(sc: SparkContext,
                    featureSize: Int,
                    featurePath: String):  CoordinateMatrix ={

    val rows = sc.textFile(featurePath).map(_.split(" "))
      .map(x=>(x.filter(g=>g.contains(":"))))
      .map(x=>(x.map(_.split(":")).map(ar => (ar(0).toInt,ar(1).toDouble))))
      .map(x=>(Vectors.sparse(featureSize,x)))

    val mat = new RowMatrix(rows)
    val simMat = mat.columnSimilarities()
    simMat
  }


  /**
    * 载入相似度矩阵
    * @param sc
    * @param simPath
    * @param featruesSize
    * @return
    */
  def loadSimMatrix(sc: SparkContext,
                    simPath: String,
                    featruesSize: Int
                   ):  SparseMatrix ={
    val entries = sc.textFile(simPath)
      .map(_.split("\\|", -1))
      .map(x=>(x(0).toInt, x(1).toInt, x(2).toDouble))
      .collect()
    val simMatrix = SparseMatrix.fromCOO(featruesSize, featruesSize, entries)
    simMatrix
  }

  /**
    * 将相似矩阵存储为文本文件
    * @param sc
    * @param savePath
    * @param mat
    */
  def saveSimMatrix(sc: SparkContext,
                    savePath: String,
                    mat: CoordinateMatrix): Unit ={
    val sim = mat
    sim.entries.map(x=>x.i+"|"+x.j+"|"+x.value).coalesce(1).saveAsTextFile(savePath)

  }

  /**
    * 根据Item 编号，从相似矩阵中获取该Item 的相似向量
    * @param simMatrix
    * @param itemNum
    * @return
    */
  def getSimVecFromMatrix(simMatrix: SparseMatrix, itemNum: Int):Array[Double] ={
    val arr1 = for (i <- 0 until itemNum) yield {
      simMatrix(i, itemNum)
    }
    val arr2 = for (i <- itemNum until simMatrix.numRows) yield {
      simMatrix(itemNum, i)
    }
    (arr1 ++ arr2).toArray
  }

  /**
    * 基于Item 相似度向量 计算推荐单个物品时的得分，输出结果按得分降序排序
    * @param sc
    * @param sim
    * @param featurePath
    * @return
    */
  def predictBySimVecWithLibSVM(sc: SparkContext,
                                sim: Array[Double],
                                featurePath: String): RDD[(String, Double)] ={
    sc.textFile(featurePath).map(_.split(" ")).map(x=>{
      val id = x(0)
      var score = 0.0
      for (i <- 1 until x.length){
        val idx = x(i).split(":")(0)
        val value = x(i).split(":")(1)
        score += value.toDouble * sim(idx.toInt)
      }
      (id,score)
    }).sortBy(_._2,false)
  }


  /**
    * 基于Item 相似度向量 计算推荐单个物品时的得分，输出结果按得分降序排序
    * @param sc
    * @param sim
    * @param featurePath
    * @return
    */
  def predictBySimVecWithDF(sc: SparkContext,
                                sim: Array[Double],
                                featurePath: String): RDD[(String, Double)] ={

    val sqlContext = new SQLContext(sc)
    sqlContext.read.parquet(featurePath).select("id","features")
      .rdd.map(x=>{
        val p = x(0).toString
        val v = x(1).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector]

        val idxs = v.toSparse.indices
        val values = v.toSparse.values
        var score = 0.0
        for (i <- 0 until idxs.length){
                score += values(i) * sim(idxs(i))
        }
        (p,score)
      })
      .sortBy(_._2,false)
  }



}

object ItemCF extends ItemCF{



  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("utils").setMaster("local[8]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val libsvmFeaturePath = "..//data//libSVMPath"
    val dfFeaturePath = "..//data//DFPath"
    val simPath = "..//data//SimPath"

    val featureSize = 3953

    testComputeSim()
    testSaveAndLoadSimMatrix()


    def testComputeSim(): Unit ={
        println("Test Compute Jaccard Sim With LibSVM ")
        val sim1 = computeJaccardSimWithLibSVM(sc,libsvmFeaturePath)
        sim1.entries.take(3).foreach(println)


        println("Test Compute Cossin Sim With LibSVM ")
        val sim2 = computeCosSimWithLibSVM(sc,featureSize,libsvmFeaturePath)
        sim2.entries.take(3).foreach(println)

        println("Test Compute Jaccard Sim With DataFrame ")
        val sim3 = computeJaccardSimWithDF(sc,dfFeaturePath)
        sim3.entries.take(3).foreach(println)

        println("Test Compute Cossin Sim With DataFrame ")
        val sim4 = computeCosSimWithDF(sc,dfFeaturePath)
        sim4.entries.take(3).foreach(println)
      }


    def testSaveAndLoadSimMatrix(): Unit ={
        val sim = computeCosSimWithLibSVM(sc,featureSize,libsvmFeaturePath)
        saveSimMatrix(sc,simPath,sim)

        println("Save The SimMatrix")

        val simLoad = loadSimMatrix(sc, simPath, featureSize)
        println(s"Load The SimMatrix. The Row Num Is ${simLoad.numRows}  The Col Num Is ${simLoad.numCols}")
      }

    def testPredict(): Unit ={
      val simMatrix = loadSimMatrix(sc, simPath, featureSize)
      println(s"Load The SimMatrix. The Row Num Is ${simMatrix.numRows}  The Col Num Is ${simMatrix.numCols}")

      val itemNum = 800
      val simVec = getSimVecFromMatrix(simMatrix,itemNum)

      println("Test Predict By SimVec With LibSVM ")
      val score1 = predictBySimVecWithLibSVM(sc, simVec, libsvmFeaturePath)
      score1.take(3).foreach(println)

      println("Test Predict By SimVec With DataFrame ")
      val score2 = predictBySimVecWithDF(sc, simVec, dfFeaturePath)
      score2.take(3).foreach(println)



    }

  }
}
