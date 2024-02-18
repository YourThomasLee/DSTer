## Phase概览

*   **涉及模块**
    *   非交互类 ReqProcessModule
    
        >   解析和merge实验参数
        >
        >   获取上游的全流量配置，分发到下游
        >
        >   获取此次检索的信息：地区信息、设备信息、query
        >
        >   设置过滤信息和过滤名单：query级别，AD级别分为winfo、uid、ideaid、trade2、titleSign、brandSign、videoSign（根据次数和间隔时间）
    
    *   非交互类 LiteSearchRedisProcessModule
    
        >   根据优先级，获取缓存tu信息
        >
        >   有些可以获取缓存如bes，有些不能Nores
    
*   **代码路径**：`feedas/framework/`

*   **执行顺序**

    >   非交互类按照index先后执行，则Req先，Lite后

    *   Module初始化 => 进程级别数据初始化
        *    [`Req::DECLEAR_MODULE_PROC_DATA`](#Req::DECLEAR_MODULE_PROC_DATA) 
        *    [`Redis::DECLEAR_MODULE_PROC_DATA`](#Redis::DECLEAR_MODULE_PROC_DATA) 
    *   初始化配置
        *   `Req: register_conf`: GO_ON
    *   初始化词典
        *   `Req: register_dict`：GO_ON
    *   QueryContext初始化 => 线程级别数据初始化
        *   [`Req::DECLEAR_MODULE_QUERY_CONTEXT`](#Req::DECLEAR_MODULE_QUERY_CONTEXT) 
        *   [`Redis::DECLEAR_MODULE_QUERY_CONTEXT`](#Redis::DECLEAR_MODULE_QUERY_CONTEXT) 
    *   非交互类串行
        *   Req: `prepare_context()`无操作  => [`handle_data()`](#Req::handle_data) => `debufpf`
        *   Redis：`prepare_context()`无操作 => [`handle_data()`](Redis:handle_data) => debugpf

## Module1. ReqProcessModule

### Req::DECLEAR_MODULE_PROC_DATA

>   模块初始化

*   创建进程级数据`ReqProcessProcData`

*   `proc_data->initialize()`: 读取配置中的间隔时间

*   `module_proc_data_macro get_proc_data()`

    >   Remix框架的base_module.cpp中定义，返回当前module的`_module_proc_data`

### Req::DECLEAR_MODULE_QUERY_CONTEXT

*   创建对象`ReqProcessQueryCtx`

*   `q_ctx->init_context()`

    *   创建`bsl::xcompool`对象`_p_pool`  => clear()

        >   bsl百度的C++基础库，替代STL。xcompool是能无限分配的pool

    *   创建`bsl::ResourcePool`对象`_p_resource_pool` => reset()

*   `q_ctx->clear`

    *   调用基类`ModBaseQueryContext`的clear：清空各项时间记录
    *   清空各集合：频控、`_shown_info*`、`_user_dislike*`的set、table、freq和时间间隔信息

### Req::handle_data

*   读取请求并反序列化 [`read_request_idl`](#Req::read_request_idl)

*   请求itp的debugpf头部，判断当前请求是否为debugpf请求 [`parse_itp_req_info(q_ctx)`](#Req::parse_itp_req_info)

*   **解析和merge实验参数**

    *   `FLAGS_bda_close_exp`：线下测试屏蔽实验参数用，线上不要开
    *   解析实验参数[`parse_exp_info`](#Req::parse_exp_info)

*   **获取上游router信息**：[`parse_router_info`](#Req::parse_router_info)

    根据上游数据`upstream_data`获取数据

*   **解析透传的asp信息到`td->aspreq_data`:** [`parse_aspreq_data`](#Req::parse_aspreq_data)

*   **生成query_sign**: 唯一标识original_query的签名

*   **计算cmatch**：src_info计算ctr_cmatch和roiq_cmatch

    >   在src_id中寻找aspreq的cmatch位置

*   **准备请求数据**: 命中无广告返回cache、bes_mt转换、bes开屏大小、尺寸过滤信息、时长过滤、建筑大师(程序化创意不进行三图投放)、日志设置、设置besrtabs

    >   Mtq: 用于程序化创意广告组合优选

#### Req::read_request_idl

*   获取线程数据`td`和asp请求数据`td.aspred_data`
*   获取当前检索环境`get_self_context`, 获取环境内存池
*   从该环境的内存池中获取remix上游请求`q_ctx->get_upstream_data()->_request`
*   获取头`(nshead_t*)(request_buf->c_str())` => 获取头长度 => 根据头长度推算消息体起始地址
*   读取流程
    *   load：`cur_ctx->_asp_req_reader->load(ns_head, body_head, body_len)` 
    *   Unpack
        *   unpack用户数据
        *   获取router传来的fengsui信息、解析上游传来的`interaction_data`、设置fengsui函数级日志采样率

#### Req::parse_itp_req_info

*   `curr_ctx->_asp_req_reader->get_debug_req`

*   有debug_req_data且size>0，则将`q_ctx->set_is_need_debugpf_info`设为true

    >   之后各模块都会用`record_debugpf_info`

*   初始化debug线程数据 `debug_thread_data_init()`

#### Req::parse_exp_info

*   从Searcher的pool中获取当前环境：`get_self_context`

    >   Lvlexp交互参数: 负责传递路由路径参数
    >
    >   Ovlexp实验参数: 负责传递全流量分层框架注册的策略配置，之后会下发到各个位置
    >
    >   yacl：实现实验参数托管。一次query结束后，参数恢复默认

*   **获取ovl全流量实验配置**：`p_ovlexp = cur_ctx->_asp_rep_reader->get_ovlexp()`

*   **保存路由信息lvlexp_info**到线程数据`q_ctx`中，后续用于bs和cs通信

*   `ym = env->get_yacl_manager` => `thread_ctx = q_ctx->get_yacl_context`

    >   用于托管实验参数

*   **设置yacl状态**

    *   从环境`q_ctx`中获取`asp_req_data`和`asp_req_idl`

    *   从`asp_req_data`获取`epvq_level`声明`epvq_buff`字符数组 => 设置ym状态

        >   此处epvq是实时观星q值的监控数据，每隔一段时间输出所有监控项目到一条监控数据中(上游router直接传来)
        >
        >   计算方式=该流量所有q值之和除以样本数计算得到，分为fast和slow两个项目
        >
        >   也有说法是激发的query的流量商业价值评估，用于query过滤

    *   从`asp_req_idl`交互数据中获取`src_list`，判断是否为有效的request

    *   日志输出version、query、src_list_size、page_no

    *   判断src_list是否为空

    *   以第一个srcid作为yacl_condition Flow的值

    *   声明srcid_buff，并用该buff设置yacl_manager状态

*   **保存全流量配置并merge实验参数**

    *   将全流量配置存入`q_ctx`，确保之后分发到下游

    *   `ym.context_merge_overlap`

        >   参数merge猜测是根据全流量实验的需求，从请求pv中获取对应的实验参数merge到yacl的con_ptr，便于后续动态调参

*   **实验id拼接**，用`_`分隔

*   **准备`itp_header`**

    如果使用了`流式检索框架fugue-lite`或`观星算子predictor依赖的gst-api`，就准备构建pb格式和idl格式的itp_header，给观星用`prepare_itp_req_header`

#### Req::parse_aspreq_data

**【获取上游数据和接口数据】**

*   从当前线程数据`q_ctx`中获取上游请求数据`aspreq_data`以及接口数据`asp_req_idl`

    >   Idl主要存储路由、wepvq、请求数等

*   从`asp_req_idl`获取上游`asp_ui_req`和`asp_info_req`

**【解析原生afd数据】**

*   从检索线程数据**解析`general_trans_data`**，存到map中

**【更新线程级数据】**

*   **解析`additional_limit`信息**

    >   在tianlu和feedproxy中有同名模块，暂时不会冲突

    获取`asp_ui_req`的`additional_limit` => 添加到线程级数据`td._tianlu_addidional_limit`中

*   **解析冷启动`cold_boot_info`信息**

    获取`asp_ui_req`的`cold_boot_info` => 添加到线程级数据`td._cold_boot_info`中

*   **解析adx请求**

    >   besafd，需要报价的请求？

    获取`asp_ui_req`的`adx_trans_req_info` => 添加到线程级数据`td.aspreq_data.adx_trans_rep`中

*   **解析`common_filter`过滤数据**

    >   如对小流量视频进行清晰度过滤、超短时长过滤 + url白名单过滤

    获取`asp_ui_req`的`common_filter_trans_data` => 添加到线程级数据`_common_filter_trans_data`中

**【解析数据源信息】**

*   调用`parse_src_info`

    从接口idl数据中获取`src_list`，遍历

    	* 将`src_id`存到`aspreq_data.src_id[]`中
    	* 判断是否mask
    	* 请求广告数
    	* 判断该请求为是wise、wap、手百还是贴吧

**【解析请求数据】**

*   **解析wepvq**：`asp_req_data.wepvq=router_info.wepvq()`

    >   wepvq: 凤巢移动搜索(wise) 预估pv质量

*   **解析search_id**：标识一次检索行为

    >   来自fmp，传来qid数组，两个32位整数拼成uint64整数

*   **初始化QueryEnv**：应用级检索，分配内存池和DataManagerQueryContext

*   **获取`src_id`**：`td.aspreq_data.src_id[0]`

    >   获取广告位标识

*   **解析`flow_tag`**

*   **解析`query_source`**：标识带url的广告位

*   **解析`original_query`**：当前请求的query

    *   贴吧列表页的original_query: 贴吧名称
    *   手百流量：上次触发广告的query

*   **解析user_ip**: 没有就解析ipv6

*   **即便有近线系统，依然下发原始sid**：`aspreq_data.original_sid = asp_ui_req->original_sid`

*   **解析region**：省和市，用整数int表示

*   **解析模板名`ui_template_name`和计费名`charge_name`**

    *   模板名：不同展现样式

        >   可能出现多个计费名对应一个模板名 => 一个模板多个人用
        >
        >   多个模板名对应一个计费名 => 一个流量按照不同抽成进行实验

    *   计费名：标识流量来源，tn=baidu(内部流量), tn=58com(58同城)

        >   到时根据计费名进行分成

*   **解析页号`page_no()`**

*   **解析上游传来的bes_level**

    >   不同媒体接入同一个cmatch，需要用bes_level表示流量质量，根据pv和charge分级
    >
    >   2不向下请求(包括作弊流量的tu)
    >
    >   不同bes_level下各src的gid=gid*bes_level
    >
    >   <img src="http://bj.bcebos.com/ibox-thumbnail98/83e1ca10f3fadf05a15cf9d2fabab025?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-29T09%3A50%3A51Z%2F1800%2F%2F71652155dcea914c8a7cc290889dd4fcb7d9180bb932aaf526afb568c814704b" style="zoom:50%;" >

*   **解析baidu_id**: 新用户访问百度产品生成的唯一id，32位字符串

*   **解析passport_user_id**: 用于不同模块交互验证 => uid

*   **解析`exper_id`**

    >   样式组id，线上的id均为0，大于0为实验组

*   **解析`gr_sid_list`**: 全局推荐的search_id?

    >   Gr: global recommendation, 全局推荐

*   **解析`user_agent`、`refer`**

*   **解析`ns_refer`、`new_cur_time`**：修复wise部分跳转来源不一致的情况、获取新的本地时间

*   **解析`ui_url`、`request_time`**

*   **解析`gps`、`cuid`(加密)、`browser_id`、`phone_id`等信息** => 解密cuid

    >   cuid：手机唯一标识存疑，手机地图内部引擎使用，百度客户端app唯一标识

*   如果是tieba数据源，**解析tieba信息**: 包括吧id、吧名、所处目录、页名、imei、idfa

    >   Imei: 国际移动装备辩识码，由15位数字组成
    >
    >   idfa: IOS独有的广告标识符

*   **解析`shown_info`**：进行频控设置和过滤名单记录,  [`parse_shown_info`](#Req::parse_shown_info)

*   **解析计费日志**字段

    >  Wi: dsp id
    >
    >  fn: 计费名
    >
    >  ctk：内容联盟，代表内容方key
    >
    >  tu：广告位id
    >
    >  adclass：请求广告类型
    >
    >  tm：媒体id
    >
    >  sdk_v：sdk版本
    >
    >  ssp：一级流量来源
    >
    >  td：托管广告位
    >
    >  ch：channel值。lu接入为0
    >
    >  nttp：新流量类型
    >
    >  Appsid：应用id
    
*   **设置bes_level**，用于之后根据不同level进行的策略分级：`parse_tu_bes_level_dict`

    *   afd打分为1（反向对照流量）和10（固定包流量）的广告位，沿用

    *   afd未打分，查阅词典`TuBesLevelDict`进行策略分级

    *   如果设置了`adjust_bes_level`，则根据epvq

    *   设置未知用户的bes_level

    *   查阅流量等级是否合规

        >   bes_level为4、5或6

*   **解析`user_dislike_ad`**: 获取用户不喜欢的广告列表

    >   时间间隔内不喜欢的cmatch(可能忽略)、广告主uid、创意idid、品牌brand、video(可能忽略)、不喜欢的item(一级行业+实体)

*   **解析`client_idfa`**

    >   idfa: IOS广告独有标识]

*   **解析`user_refresh_info`**

    >   刷新动作refresh_state: 0下拉，1第1次进入，2中间提示等
    >
    >   刷新次数refresh_count：可能根据session刷新重置
    >
    >   请求刷新的次数refresh_req_num

*   **解析手百刷新吸顶状态**

*   **解析设备id**：deviceid、idfa、imei、oaid、caid（广告归因）

*   **判断计费串是否增加CK字段**：反作弊策略（即过滤无小电机）

*   **解析被限制的广告id**，同时把广告对应的uid放入`userid_freq_control_set`

    >   过滤优先级：强制过滤名单(forced) > 黑名单(blacklist) > 域名白名单(whitelist) >物料白名单(rules_exclude_patterns) > 域名级规则(domainlist) > 通用过滤规则(default)

*   **解析爱奇艺视频数据**: `parse_iqiyi_data`、`parse_iqiyi_url_sign_filter`

    剧目album_id、合集channel、视频长度、视频资源url_sign

*   解析直播间数据

*   **解析mt样式黑名单**：单pv多src，获取多流量的黑名单取交集传给feedproxy，兜底过滤时使用各自流量

*   **解析飘蓝词、mt白名单、query所属行业、点击广告信息**

*   **解析流量归属(兴趣领域)**：手百新闻、手百视频、手百视频详情(如果开启视频特征)、feed视频标签、解析当前feed流量种类、解析bes文章内容(如果开启页面兴趣)、解析bes文章字段(id、作者id、文章来源)、解析汇川小说

*   **解析手百经纬度、deeplink信息、os版本、ssp保护(媒体保护白名单)、根据tu增加媒体id**

    >   Deeplink: 从h5或app跳到另一个app指定页面

*   **解析计费监控字段**：channel_id、安卓id、dc_cuid解码后公司等

*   **解析多样式大小、设置bes样式组信息、解析危险行业(获取允许行业)**

*   **解析闪投模块页面信息、新用户**

    ...

*   **解析激励视频流量的appid**

*   **解析激励视频的视频时长和视频大小**：获取词典、获取src_id和app_id，根据两者查询视频文件，解析视频时长和视频大小

*   **本地词典扩词，用于结构化触发**：增加汇川游戏检索词 + 百度阅读传递的标题、分类、二级分类

*   **解析知道信息、地点信息类型、敏感词`sensitive_package`、timeout、供应商supid、解析最小cpm、dmp信息、adx黑名单、预算级别**

*   **解析GD相关参数**

    ...

*   **初始化as漏斗信息**`feed_filter`

#### Req::parse_shown_info

>   Shown_info：afd记录最近10次pv，保留8h
>
>   也有说法说过期时间是24小时

*   **配置文件**：`frep_control.conf`

*   **一次pv的数据结构**

    >   bw: bid word
    >
    >   ts: title sign
    >
    >   bs: brand sign
    >
    >   pl: plan id
    >
    >   cm: cmatch 
    >
    >   vs: video sign
    >
    >   tr2: trade2
    >
    >   fmid: format?

    <img src="http://bj.bcebos.com/ibox-thumbnail98/55abb8dda800a6e66d35ecf7e0eefe2e?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-07-30T03%3A41%3A13Z%2F1800%2F%2Ff4ee00253899930c58bd60cbf162e94d964e6f5d620e014d687c2d8e657458ea" style="zoom: 50%;" >

**【阈值设置】**

>   Dedup_freq用于控制出现次数多于阈值的广告
>
>   time_interval是频控时间，xx时间内出现过，则过滤

*   频控信息：包括时间间隔`time_interval`、`user_dedup_freq`、`winfo_dedup_freq`、`ideaid_dedup`、`query_dedup`、`query_time_interval`、title_sign、brand_sign、vedio_sign
*   冷启动要降低整体shown_info时间频控
*   从general_trans_data中获取shown info 
*   赋值dedup_set和dedup_table

**【进行过滤】**

>   time(NULL)用于获取当前时间

*   **query过滤**：`top_pv_freq_dedup`

    *   记录每个pv的时间`time_list`
    *   统计query出现次数和时间间隔，`次数多`或`次数多且间隔短`的query被过滤（插入到，逐渐完善要过滤的query集合

*   **ad过滤**

    >   Wdid = winfoid；winfo是广告关键字段；idea是广告创意

    *   统计winfo的出现次数和间隔时间，过滤次数多且间隔的winfo
    *   统计uid(广告主)的出现次数和间隔时间，过滤次数多或次数多且间隔短
    *   统计时间间隔内二级行业广告的出现次数，过滤次数多且属于要管控的二级行业`is_in_control_trade2` => 比其他频控信息多存储的步骤
    *   统计ideaid，过滤次数多或次数多且间隔短
    *   依次统计标题titile sign、品牌brand sign、视频video sign、csid，过滤次数多且间隔短的

**【信息统计】**

*   plist(产品信息?) 记录到adinfo

*   获取当前cmatch，将该次pv的planid存入该cmatch对应map

    >   `showninfo_click_planid_map`key是cmatch，value是存储planid的set

*   若`last_cpm`的pv时间比此次pv时间早，则用此次pv的cpm更新`last_cpm` 尾部，否则需要执行插入

## ReqProcessQueryContext

### 作用

*   存储频控信息、`_shown_info*`、`_user_dislike*`的set、table、freq和时间间隔信息

*   封装对这些数据的set和get操作

### 友元类

*   ReqData
*   GoldengateData

## Module2. LiteSearchRedisProcessModule

### Redis::DECLEAR_MODULE_PROC_DATA

*   创建进程级数据`LiteSearchRedisProcessProcData`

*   `proc_data->initialize()`：无操作

*   `module_proc_data_macro get_proc_data()`

    >   Remix框架的base_module.cpp中定义，返回当前module的`_module_proc_data`

### Redis::DECLEAR_MODULE_QUERY_CONTEXT

*   创建对象`LiteSearchRedisProcessQueryCtx`
*   `q_ctx->clear()`
    *   调用基类`ModBaseQueryContext`的clear：清空各项时间记录
    *   重置各种状态：redis状态、广告数量、proxy广告数量、清空_tu_group
*   返回0

### Redis:handle_data

*   获取QueryContext`curr_ctx`

*   `curr_ctx->_is_request_redis_status=1`

*   如果需要缓存，则在线程级数据中创建缓存信息`redis_key`

    >   优先级cuid>baiduid>deviceid

*   如果redis是空的，函数直接返回

*   **缓存获取**，存入redis_key

    *   bes优先级：根据asp传来tu，获取缓存tu信息

        >   短期内同一个广告位返回相同广告

    *   nores优先级(无广告)：将执行全流程广告检索

    *   其余优先级: 获取联合缓存tu_group

*   初始化redis_client_manager和redis_client
*   根据key获取client和response