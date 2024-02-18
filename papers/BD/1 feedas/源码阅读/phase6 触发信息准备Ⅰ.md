## Phase概览

*   **涉及模块**
    *   交互类：GoldenGateProcessModule
    *   非交互类：UserEmbeddingProcessModule
    *   非交互类：RedisProcessModule
    *   非交互类：XboxCenterProcessModule

*   **代码路径**：`feedas/framework/`

## 执行顺序

*   **交互类前期**

    每个非交互类的执行顺序`create_rpc_channel` => `prepare_request` => `send_request`

|   Module   | prepare_request                                              | handle_response                                              | 作用                                                         |
| :--------: | :----------------------------------------------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| GoldenGate | 主要函数：`make_goldengate_request`<br> ① 从**asp数据**中获取：sid、flow_tag、original_query、baiduid、cuid、ua、refer、tieba标题、`ctr_cmatch`、感兴趣的图和视频等<br>②  从**用户信息**中获取：short_attention(ums)、long_term_fea(upin)、shoubai_mt(uc)、人群包信息(upin)、location(dmp)、business_tag(dmp)、tieba实时特征和历史特征(ums)、实时广告及点展行为数据(intentservice)等<br>③ 设置**频控信息**（query级别），为金门提供query精选依据：upin_query_dedup_set和shown_info_query_dedup_set<br>④ 添加**已有query信息**：汇川中间页、贴吧标题、dmp的ei_query(entity intention)<br>⑤ **小程序信息**：页面id、小程序信息流的一级和二级分类<br>⑥ **pat**需as下发字段：os_type、device_type、屏幕长宽等 | 主要函数: `parse_goldengate_response`<br>① 获取金门从不同分支branch拉取整合而成的**query_list**：query、branch、barnch_rank、branch_weight、intentq、epvq、cpm、trade_id、score、pv<br>② 解析**query_mining_list**: 参数同上<br>③ 解析**kg意图词kg_list**：query签名、branch分支、score、intentq、pvr(有效检索比率,epv/pv)，intent_level<br>④ 其余列表：激励视频列表、相似人群包、otpa人群包(广告主定向人群包）、搜索词、搜索词改写(eureka)、兴趣列表、知识广告等<br/> | 1. 单query -> 多query<br>2. 无query->有query，一般是feed流量query化t包括query生成和改写)：输入此次pv信息及用户信息，从金门获取query_list |

>   *   ctr_cmatch: 请需要请求观星ctrq的cmatch
>
>   *   上述upin_dedup_set现在是由usercenter创建
>   *   涉及的query q值
>       *   intentq: 当前query和用户的相关性
>       *   epvq：query召回广告的预估点击率，衡量query的流量商业价值
>       *   Deepintentq: 基于dnn的intentq
>
>   <img src="http://bj.bcebos.com/ibox-thumbnail98/c9bedd2d8ff661abf7cb164459001237?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-05T02%3A41%3A26Z%2F1800%2F%2Fa2c49735572dd8cd3ac93c74043c786290173a0a21cf57ffa496214a3d9e0ca3">

*   **非交互类**

    基本顺序为：prepare_context => `handle_data` => debugpf

|               | handle_data                                                  | 作用                                                         |
| :------------ | :----------------------------------------------------------- | :----------------------------------------------------------- |
| UserEmbedding | ① 打包观星请求`pack_sess_request_data`: 当前session中ums和upin的数据<br>② 异步接收返回并解析观星结果`embd_result`: 根据`predictor.conf`获取对应q值feeduserq、feedannq、ocpcuserq、feedroiuserq等<br>③ 请求item2item的xbox，获取用户相关广告: article_nid、 realtime_idea_id、idea_id_profile、idea_id_priority<br/> | ① 请求观星获取用户向量<br>② 获取xbox相关广告队列`user_related_adv_res_list`的user_vector，写入观星获取的feeduserq信息，待之后进行相关广告获取 |
| RedisProcess  | 1. 获取adv缓存:ideaid、unitid<br>2. 获取ums缓存：short_attention、attention_statistic、感兴趣的一二级类别、感兴趣图和视频、attention_dislike<br>3. 获取详情页面标题: 获取高质广告<br>4. 获取zhuida缓存和元信息 | 请求redis，获取idea_id(请求feedproxy需要）、unit_id、ums缓存、高质广告等 |
| XboxCenter    | ① FeedOnlineInterest：传图upin的session搜索记录和历史搜索记录，返回好看兴趣点<br>② Item2item：获取用户相关的广告队列<br>③ 其余：PositionUserInfo、Boostpv、AcgCreditScore | 请求xbox数据，获取好看兴趣点、用户相关广告向量等             |

>   *   zhuida：用户点击一条视频后，系统持续推荐多条相似视频，但用户不再感兴趣的情况 => 推荐退场不及时
>   *   冷启动：对重点客户的新建单元进行扶持，帮助模型快速收敛
>   *   boost扶持：广告投放期间，某些场景有起量诉求

*   **交互类后期**

    基本顺序为：wait_response => handle_response => close_rpc_channel

## GoldengateProcessModule

### prepare_request

*   `make_goldengate_request`

    略过信息：feedpat部分和小程序部分