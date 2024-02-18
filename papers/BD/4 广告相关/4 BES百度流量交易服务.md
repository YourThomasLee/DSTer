>   Ref: [BES](http://learn.baidu.com/pages/index.html#/video/?courseId=5881&elementId=cf92be9b-b1b5-4bd3-8f7b-2486ec0e1669&userId=6849593&groupId=6092&curPlayIndex=0)
>
>   Bes: Baidu Exchange Service

## 百度联盟的位置

*   百度相当于中介，一方连接出卖广告位的流量方，另一方连接要买广告位的广告主

## 百度的两种广告投放模式

*   **计费方式**
    *   点击
    *   展现：有一种广告主不一定需要点击，而只是通过展现来提高品牌知名度 => 可以转化为线下零售量
    *   按交易计费
    *   换量方式：APP展示百度的广告，百度给APP高位

<img src="http://bj.bcebos.com/ibox-thumbnail98/fa2d5fdea72946ec769bc3d97c19c5e5?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T11%3A18%3A11Z%2F1800%2F%2F8c00b945615efc43466d894c12d8a1e0de6f078d8a4a0788b0181a017740e569" style="zoom:50%;" >

### 传统AdNetwork（网盟广告）售卖模式

*   百度根据日志进行付费
*   在这种模式中，网站主对于百度来说是更重要的（类似房源）

<img src="http://bj.bcebos.com/ibox-thumbnail98/c824be98c90ec729636d8b3ee81ff1ce?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T11%3A31%3A15Z%2F1800%2F%2F88b35f7e2994331570084c1d2ebf9b6036bd6d287573979cc9615786b3087aa7" style="zoom:50%;" >

### AdExchange（程序化购买）售卖模式

*   **背景**

    原始模式：直接出卖位置，百度为这个位置选择最匹配的广告主

*   **新模式**

    *   公开竞价：广告位卖给百度，百度把这个位置公开对外进行拍卖，接入多个DSP（包括百度自己），各个平台根据自己对当前浏览该页面的用户进行评 估，猜测这个用户的点击率或这个广告位对自己的价值，据此进行竞价
    *   私有竞价：百度DSP优先内部沟通，直接买断该广告位
    *   优先交易：广告主与媒体直接有相关协议，百度优先把该媒体的某个广告位给到某个广告主
    *   保量交易GD：Guaranteed Delivery，某个广告主直接指定在哪个流量（体育或新闻）等有多少次展现 => 保证展现次数

*   **优先级**

    广告位优先给到GD广告

*   **为什么引入外部DSP**

    从媒体角度考虑，只与百度DSP，每次只能看百度的出价来分钱，引入竞争，使得媒体的收入增多 => 选择出价最高的DSP进行广告位出卖

<img src="http://bj.bcebos.com/ibox-thumbnail98/51828665d22d5ed0306b2b7916c6127d?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T11%3A35%3A40Z%2F1800%2F%2Fc7d1cce620f8be87fb52536821cb5f9981b37b7263370b80a28346b6b43ca529" style="zoom:50%;" >

## 计价机制

*   **程序化购买到底买了什么**

    当前用户在当前页面某个广告位的一次广告展现机会

*   **出价出的是什么**

    CPM千次展现价格 or CPC模式

*   **实际交易价**

    第二高出价价格 + 1分

    >   放任大家随便出价，价格波动

*   **DSP怎么赚钱**

    广告主支付投放平台DSP广告费用，DSP购买展现

    >   如一次点击1块，而DSP在竞价平台上买的展现一次8毛 => 则DSP可以赚差价

*   **BES怎么赚钱**

    BES将流量卖给DSP，根据CPM跟流量分成，分得的钱就是利润

## 名词解释

*   **信息流广告**：发现不了是广告，比如微博、微信
*   **RTB**：广告实时竞价。每次广告展现的拍卖方式。加高者胜。

*   **ADX(ad exchange)**：广告交易平台

*   **SSP**：Seller Side Platform 流量供应平台，用于管理所有接入的流量

*   **DSP**：Demand Side Platform 广告供应平台，用于检索、排序、竞价每次impression，并返回广告

    >   用于存储广告物料，检索广告

*   **BES**：Baidu Exchange Service, 百度流量交易平台

    >   就是百度的adx

## 资源体系

<img src="http://bj.bcebos.com/ibox-thumbnail98/b09ce5b347f26257c3afb90761a1bfd4?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T12%3A01%3A01Z%2F1800%2F%2F224ae9f6637faa6a1fc844bff42fb80ca190f1fbcebb5006d2da22045413de07">

*   SSP - BES - DSP

    *   北斗：传统联盟广告，直接投放空的广告位

    *   Baidu DSP: 广告主提出特别具体的人群定位需求

    *   移动 DSP：移动端

    *   链接单元DSP：学习当前用户的感兴趣的词 => 此类广告

        >   ![image-20210726200422808](https://gitee.com/WIN0624/document/raw/master/img/image-20210726200422808.png)

## 广告竞价流程

<img src="http://bj.bcebos.com/ibox-thumbnail98/5e71e9f7b13953664e330a780da5d75a?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T12%3A07%3A41Z%2F1800%2F%2F37a37a98698c2bdbd879ab2c5171a9d17ad050c0df1247d4da02d1aa7ca34266" style="zoom:50%;" >

*   用户浏览一个网站
*   网站包含js代码，向流量销售平台发起请求（SSP）
*   流量销售平台向一家或多家ADX发起广告请求
*   ADX平台向多家流量购买方DSP发送竞价请求
*   流量购买方分析请求，决定此次展现的出价，将广告和竞价返回到ADX

*   ADX收集所有购买方的竞价结果，返回出价最高的购买方广告
*   SSP返回广告
*   用户看到一个展示广告

## 展示广告检索系统基础架构

*   **DSP的出价依据**

    *   主要工具：DMP
    *   作用1：分析用户的兴趣（平时经常搜索什么）
    *   作用2：分析当前流量的质量（所处页面、页面中位置等）

*   **DSP的具体出价**

    *   模型：CTR

        >   广告热门度、广告质量、用户兴趣度 => 根据用户兴趣来出价

<img src="http://bj.bcebos.com/ibox-thumbnail98/7bf72c1487bb50412b37eec29120812d?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T12%3A08%3A38Z%2F1800%2F%2F86a16ca7eb07fb33094c171d78aba34a3fea8e1ea720355d7206e60071d23c40" style="zoom: 33%;" >

*   **总体架构**

    *   BFP: 外部流量统一入口，用于管理广告位置 

        >   即说明这个地方可以出广告，如果请求到了就出广告，请求不到就出自定义的图片

    *   ADX

        *   server：把上游流量分发到所有广告投放端 + 把下游提交的报价请求进行排序取最高，返回流量平台

    *   左下方板块

        *   NOVA: 传统广告
        *   UBMC：用于广告物料保存
        *   prediction service: 根据广告昨天点击情况预估今天是否会点击
        *   render：广告文本渲染为HTML
        *   budget:  广告不是无限对外投放，有一定预算。预算花完就不再为其竞价，将其投放 => 用于对广告进行实时计费，发送信号

    <img src="http://bj.bcebos.com/ibox-thumbnail98/b57a29893d79e7af94f2676791b51468?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T12%3A12%3A35Z%2F1800%2F%2Fc4afc188eba28e6df3baf16e4f285ac2d4b21928377437febc73c33b64502157">

    <img src="http://bj.bcebos.com/ibox-thumbnail98/1e88e876c9dd0da5a458a7a508c750e4?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T12%3A22%3A50Z%2F1800%2F%2Fbb5d0008d2377464d2900aa122c41e2d24e484935fb8d9cc13d40e42cf056120">

### ADX

**【WIN notice】**

*   RTB下，不是按点击计费，而是按展现计费 => 记录当时的竞价

<img src="http://bj.bcebos.com/ibox-thumbnail98/f54e0cd6a1f2b0425eaab27f95a4b61b?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-26T12%3A24%3A01Z%2F1800%2F%2F03ad09fcd2380d5a4bc3b100f86db192c7c3004c388b835798556ffb8761059d" style="zoom:50%;" >

