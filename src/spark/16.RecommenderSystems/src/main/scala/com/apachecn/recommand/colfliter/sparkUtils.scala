package com.apachecn.recommand.colfliter

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ych on 2018/9/20.
  */
object sparkUtils {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("utils").setMaster("local")
    val sc = new SparkContext(conf)

  }

}
