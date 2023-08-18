2.1.0 工程基础

## 1. spark

作为大数据计算框架MapReduce的继任者，Spark具备以下优势特性：![image.png](:/b83ac2a754ee48e19dcefd9eeaa8180b)

* 高效：不同于MapReduce将中间计算结果放入磁盘中，Spark采用内存存储中间计算结果，减少了迭代运算的磁盘IO，并通过并行计算DAG图的优化，减少了不同任务之间的依赖，降低了延迟等待时间。内存计算下，Spark 比 MapReduce 快100倍
* 易用：不同于MapReduce仅支持Map和Reduce两种编程算子，Spark提供了超过80种不同的Transformation和Action算子，如map,reduce,filter,groupByKey,sortByKey,foreach等，并且采用函数式编程风格，实现相同的功能需要的代码量极大缩小。![image.png](:/0cafaeeb09a44d35ada8014cdd299274)
* 通用：Spark提供了统一的解决方案。Spark可以用于批处理、交互式查询（Spark SQL）、实时流处理（Spark Streaming）、机器学习（Spark MLlib）和图计算（GraphX）。这些不同类型的处理都可以在同一个应用中无缝使用。这对于企业应用来说，就可使用一个平台来进行不同的工程实现，减少了人力开发和平台部署成本。![image.png](:/c59a64bd1f3742189f4ccd1980213b76)
* 兼容：Spark能够跟很多开源工程兼容使用。如Spark可以使用Hadoop的YARN和Apache Mesos作为它的资源管理和调度器，并且Spark可以读取多种数据源，如HDFS、HBase、MySQL等。


任务概念层

![image.png](:/7150d48549f244dc9663a98151ba4f59)

RDD：是弹性分布式数据集（Resilient Distributed Dataset）的简称，是分布式内存的一个抽象概念，提供了一种高度受限的共享内存模型。它是记录的只读分区集合，代表一个不可变、可分区、里面的元素可并行计算的集合。

DAG：是Directed Acyclic Graph（有向无环图）的简称，反映RDD之间的依赖关系。

Driver Program：控制程序，负责为Application构建DAG图。

Cluster Manager：集群资源管理中心，负责分配计算资源。

Worker Node：工作节点，负责完成具体计算。

Executor：是运行在工作节点（Worker Node）上的一个进程，负责运行Task，并为应用程序存储数据。

Application：用户编写的Spark应用程序，一个Application包含多个Job。

Job：作业，一个Job包含多个RDD及作用于相应RDD上的各种操作。

Stage：阶段，是作业的基本调度单位，一个作业会分为多组任务，每组任务被称为“阶段”。

Task：任务，运行在Executor上的工作单元，是Executor中的一个线程。

总结：Application由多个Job组成，Job由多个Stage组成，Stage由多个Task组成。Stage是作业调度的基本单位。


载体概念层

Spark集群由Driver, Cluster Manager（Standalone,Yarn 或 Mesos），以及Worker Node组成。对于每个Spark应用程序，Worker Node上存在一个Executor进程，Executor进程中包括多个Task线程。

![image.png](:/4979fa41c4664fc998a6f528885371d9)

RDD是Spark的基本数据结构，一般有两种方式可以创建RDD，第一种是读取文件中的数据生成RDD，第二种则是通过将内存中的对象并行化得到RDD。

```scala
//通过读取文件生成RDD
val  rdd = sc.textFile("hdfs://hans/data_warehouse/test/data")

//通过将内存中的对象并行化得到RDD
val num = Array(1,2,3,4,5)
val rdd = sc.parallelize(num)
//或者 val rdd = sc.makeRDD(num)
```

RDD的操作有两种类型，即Transformation操作和Action操作。转换操作是从已经存在的RDD创建一个新的RDD，而行动操作是在RDD上进行计算后返回结果到 Driver。Transformation操作都具有 Lazy 特性，即 Spark 不会立刻进行实际的计算，只会记录执行的轨迹，只有触发Action操作的时候，它才会根据 DAG 图真正执行。

![image.png](:/6b8def3e3b614463b3a76495d474ce7f)

RDD之间的依赖关系有两种类型，即窄依赖和宽依赖。窄依赖时，父RDD的分区和子RDD的分区的关系是一对一或者多对一的关系。而宽依赖时，父RDD的分区和自RDD的分区是一对多或者多对多的关系。

宽依赖关系相关的操作一般具有shuffle过程，即通过一个Patitioner函数将父RDD中每个分区上key不同的记录分发到不同的子RDD分区。![image.png](:/4bef4414bdfa463c9d86ce56534d5c93)

依赖关系确定了DAG切分成Stage的方式。切割规则：从后往前，遇到宽依赖就切割Stage。

RDD之间的依赖关系形成一个DAG有向无环图，DAG会提交给DAGScheduler，DAGScheduler会把DAG划分成相互依赖的多个stage，划分stage的依据就是RDD之间的宽窄依赖。遇到宽依赖就划分stage,每个stage包含一个或多个task任务。然后将这些task以taskSet的形式提交给TaskScheduler运行。


执行流程

1. Application首先被Driver构建DAG图并分解成Stage。
2. 然后Driver向Cluster Manager申请资源。
3. Cluster Manager向某些Work Node发送征召信号。
4. 被征召的Work Node启动Executor进程响应征召，并向Driver申请任务。
5. Driver分配Task给Work Node。
6. Executor以Stage为单位执行Task，期间Driver进行监控。
7. Driver收到Executor任务完成的信号后向Cluster Manager发送注销信号。
8. Cluster Manager向Work Node发送释放资源信号。
9. Work Node对应Executor停止运行。![image.png](:/cd95c2642deb4ddd8e7058a4c8f5b24b)


id: 4aaaf8dcdd6c4fad81ecf91da1079b6d
parent_id: 31604b05516a401aa614a8b7349c3ef2
created_time: 2023-03-25T10:09:45.355Z
updated_time: 2023-03-31T12:47:38.134Z
is_conflict: 0
latitude: 0.00000000
longitude: 0.00000000
altitude: 0.0000
author: 
source_url: 
is_todo: 0
todo_due: 0
todo_completed: 0
source: joplin-desktop
source_application: net.cozic.joplin-desktop
application_data: 
order: 1680266857982
user_created_time: 2023-03-25T10:09:45.355Z
user_updated_time: 2023-03-31T12:47:38.134Z
encryption_cipher_text: 
encryption_applied: 0
markup_language: 1
is_shared: 0
share_id: 
conflict_original_id: 
master_key_id: 
user_data: 
type_: 1