## Phase概览

*   **涉及模块**
    *   非交互类：FeedbesXboxModule
    *   非交互类：UnionidProcessModule

    *   非交互类：FeedUserXboxCenterMocule
    *   交互类Pdrpc：UasProcessModule
    *   交互类Pdrpc：UserCenterProcessModule
    *   交互类Raw: UpinProcessModule
    *   ~~交互类Pdrpc：KaiwuProcessModule~~ [弃用]
    *   交互类Pdrpc：LiteKaiwuProcessModule
    *   交互类Pdrpc：UmsProcessModule
    *   交互类Pdrpc：LiteGoldengateProcessModule
    *   交互类Pdrpc：IntentServiceProcessModule
    *   交互类Pdrpc：RtaserverProcessModule

*   **代码路径**：`feedas/framework/`

    >   liteGoldenGate: tools/src/


## 执行顺序

>   初始化按照index先后执行
>
>   实际执行，先交互，后非交互

*   **Module初始化 => 进程级别数据初始化**

    *    所有类按顺序进行进程数据初始化 `DECLEAR_MODULE_PROC_DATA`

         >   前三个非交互 & uas & usercenter：return 0
         >
         >   upin：读取配置文件

*   **QueryContext初始化 => 线程级别数据初始化**

    *   基本都执行`clear()`

    *   upin清空频控set和table

        >   清空后执行到phase2就完成了新频控信息的建立

*   **交互类前期**

    *   每个非交互类的执行顺序`create_rpc_channel` => `prepare_request` => `send_request`

    *   **功能概览**

| Module         | prepare_request                                              | Handle_response                                              | 作用                                                         |
| :------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Uas            | 设置设备id、src_id、search_id、全流量实验参数以及要请求的用户属性 | 获取用户自然属性及置信度，存入upin人群包：年龄、年龄v2、性别、用户画像、兴趣、意图 | 获取用户的自然属性：年龄、性别、实时兴趣、意图、星座、行业、资产、教育水平、消费、工作、婚姻情况。<br>优先级高于upin获取的自然属性 |
| UserCenter     | 设置<br>1. 设备信息：设备id、src_list、cmd<br>2. 频控信息：设置多种或一种厂商频控bes_freq_type、设置要进行频控的cmatch<br>3. 历史信息: 不喜欢的cmatch列表、sess_cmatch和show_cmatch、<br>4. 其余信息: bes_level、烽燧日志（也许是为了记录请求耗时）、要请求的漏斗信息`need_funnel_info` | 1. 统计每个广告位的曝光过的广告及点击情况<br>2. 统计asp展示过的广告<br>3. 获取xbox离线信息<br>4. 获取频控广告集合以及行业集合（dedup_set和dedup_table） | 获取有频控限制的cmacth list，请求这些cmatch过去8小时（session级别）的真实曝光情况(session)，得出各个维度需要过滤的集合 |
| Upin           | 1. 设置search_id<br>2. 设置请求来源相关信息: ip、省市、refer、original_query<br>3. 根据流量来源设置字段：baiduid，wise还有代理ua和gps<br>4. 设置需要频控的src和cmatch | 1. 解析人群定向：用户自然属性,与uas一致（uas优先级更高）<br>2. 历史搜索记录：session维度(8小时), term_list、adurl、asurl<br>3. 解析asp日志：profile维度(72小时)<br>4. 对wise流量进一步解析定位、常驻点以及query | 获取用户历史搜索信息和短期兴趣                               |
| LiteKaiwu      | 1. 基本信息：search_id、client_id、request_type<br>2. 设备情况：cuid、baiduid、tiebaidfa、os_id、passport_user_id、device_id | 获取相似人群集合`looklike_crowds`和闪投用户信息`pa_user_profile` | 网盟流量走LiteKaiwu,获取人群定向信息                         |
| Ums            | 1. 按照流量类型设置要求的字段：app是cuid+uid, wise是baiduid+uid<br>2. 设置请求信息：长期兴趣元信息、图片列表、视频列表、好看兴趣、贴吧兴趣 | 解析左侧栏请求的各项信息，此处还增加了对不感兴趣的解析       | 根据用户自然浏览时产生的attention信息：1. 短期兴趣attention_short的近期点击（好看、贴吧等）<br>2. attention_dislike<br>3. attention_statistics:attention维度的点展<br>4. 高端人群long term的标签（用于后续过滤） |
| LiteGoldengate | GO_ON                                                        | GO_ON                                                        | 默认不请求该模块                                             |
| IntentService  | ① 填充flow_info: ip、os_id、flow_type、baiduid、imei、idfa等<br>② 发出idm请求<br>③ 处理厂商预取<br>④ 处理实体请求、投资意愿请求、行为数据、茧房数据（曝光广告）、改写请求和dmp请求 | ① 解析广告曝光数据：uc_als<br>② 解析意愿数据：用户意图query、意图广告、意图兴趣<br>③ 解析行为数据：广告点击、展示和播放时数据 + query列表 + dislike数据（广告、实体、品牌和一级行业）<br>④ 解析茧房数据：屏蔽曝光数过半的二级行业<br>⑤ bes流量：获取改写的query<br>⑥ 获取dmp数据：实体意图query、用户标签、用户地点、app信息、相似人群等 | 请求意图中台获取：<br>1. 个性化数据: 行为+画像+标签<br>2. 获取商业意图，如对feed流量获取字面query、结构化意图（知识图谱）、embedding |
| Rtaserver      | ① 根据支持预取的客户list，设置交互信息：url、token、缓存时间等<br>② 设置请求权限<br>③ 设置当前流量请求的src_id、cmatch、设备信息以及媒体行业 | 判断请求是否成功，将返回结果存入线程数据q_ctx中              | 请求允许rta预取的客户，获取预取结果                          |

*   **非交互类**
    *   基本顺序为：prepare_context => `handle_data` => debugpf
    *   有两个涉及xbox的类，主要看`fill_request`和`parse_response`

|                    | handle_data                                                  | 作用                                                         |
| :----------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| FeedbesXbox        | 注册xbox请求，获取BES流量用户特征（profile和安装的app）: BesProfileXboxRequest、SdkApplistParllelRequest（安装app）、BesProfileListParllelRequest（历史query Profile） | 获取bes流量的用户特征<br>profile是用户过去72小时在cmatch、ideaid、mt等维度的访问行为 |
| Unionid            | ① 获取unionid<br>② 获取RTA预取词表，请求其中每个url是否接收此设备 => 只等待cache信息(蓝色1) | ① unionid用于实现跨小程序的用户区分<br>② rta预取结果之后下发到feedproxy，过滤adplus的rta客户 |
| FeedUserXboxCenter | BesProfile、安装app、历史搜索query、xbox获取广告频控ideaid和高品质广告、长期特征、视频特征、品牌频控等<br>有些部分用到upin，是因为upin有些数据不能直接用于触发，需要经过xbox转化 | 获取用户行为数据                                             |

>   RTA补充
>
>   <img src="http://bj.bcebos.com/ibox-thumbnail98/469ab3366a8cab1302d3d7bc65962b6c?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-02T06%3A45%3A03Z%2F1800%2F%2Ff5fb40017298ffa11358439272a99c34f13883bab01d4a6eed5a52d7f57c6ba7" style="zoom:35%;" >
>
>   <img src="http://bj.bcebos.com/ibox-thumbnail98/e9e4c13823cd47a00a8945c68bef8218?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-04T06%3A29%3A34Z%2F1800%2F%2F57e3f110adc2523b00910f3a92e982bc6fc5c860291678ba481f51842484d0cd">

*   **交互类后期**
    *   基本顺序为：wait_response => handle_response => close_rpc_channel

## UasProcessModule

### prepare_request

*   **make_uas_request**

    >   uas_request, 主要从asp_req中来

    *   设置client_name（feedas-wise）和password

    *   设置插件名`Merge`

    *   检查设备id、百度id是否完备

    *   添加`flow_type_sid`：`app/wap(类似uc浏览器访问)/pc/unknow`-`src_id`-`search_id`

        >   标明哪个类型的search

    *   获取上游下发的全流量实验参数：如`abtest_params`

        >   也是要写入请求的，`uas_request->add_abtest_params()`

    *   设置请求字段`request_item_bd`：年龄、性别、实时兴趣、意图、星座、行业、资产、教育水平、消费、职业类别、婚姻情况

*   **add_request**

    发送request

### Handle_response

*   mri_timer记录解析返回的整体耗时

*   **get_response**

*   获取controller：也保存了请求情况

*   记录uas_ip

*   交互信息添加到烽燧日志

*   设置`uas_valid=true`，之后可以使用uas数据

*   **parse_uas_response**

    *   判断是否成功：`status` + item>0 + cookie_level_dt_size

        >   get_uas_response_item: userid > cuid > baiduid > device_id

    *   获取upin的wise_dt，之后会将`uas_dt_res`的数据存入

        >   dt: demographic targeting 人群定向

    *   解析自然属性以及对应的置信度confidence/weight: 年龄、性别、用户画像list、存储大于阈值的兴趣和意图（单一source和多source处理方式不同）、收入水平、所在行业、星座、应用列表、婚姻状况、设备、年龄点、资产状况、教育水平、职业类别、消费水平、收入水平、视频观看信息

        >   置信度低于阈值，则不使用
        >
        >   ref：[联盟用户画像](http://wiki.baidu.com/pages/viewpage.action?pageId=1043202130)

    *   根据xbuiltin，将appname转换为entity，输出diff

## UserCenter

### prepare_request

*   **Make_usercenter_request**

    >   请求对象`usercenter_request`

    *   设置qid、src_list(广告位)、cmd、flowtype、设备信息cuid、baiduid、device_id

        >   cmd：0为默认、1为高质(手百相关)、2为低质(非手百相关一律低质)
        >
        >   三种状态执行不同策略插件

    *   **设置对应的cmatch`usercenter_request->add_freq_cmatch_list`**

        >   freq_type对应的expand不能为空'*'

        *   **多重频控：网盟流量 => 设置厂商频控`parse_changshang_freq_control`**

            >   词典：changshang_freq_control_dict.txt

            *   查看tu是否有效
            *   填入信息到`curr_ctx`：bes_freq_type频控类型、扩充cmatch列表、appsid列表、tu列表
            *   设置频控时间间隔到`curr_ctx`：整体时间间隔、userid、ideaid、planid、winfo、title、query
            *   若网盟流量**厂商频控词典**查询成功，遍历`changshang_cmatch_expand_list`，添加每个cmatch信息到`usercenter_request`

        *   **没有查询成功 => 单一频控：设置tu频控`parse_tu_expand_freq_control`**

            >   词典：tu_expand_freq_control_dict.txt

            *   查看tu是否有效
            *   获取cmatch_expand、appsid_expand、tu_expand
            *   设置信息到`curr_ctx`
                *   tu_expand_bes_freq_type：优先级 appsid_expand > tu_expand
                *   设置当前tu的control_id: winfo或idea或plan或user
                *   设置time_interval
            *   **tu词典查询成功，遍历`tu_expand_cmatch_expand_list`**

        *   如果查询失败，先添加当前cmatch，再遍历src_list, 获取src对应的cmatch加入到request

    *   在配置的showlist中获取要展示的广告位cmatch：`add_show_cmatch_list`和`add_sess_cmatch_list`

    *   bes_freq_select：更改freq_type

        *   规则1：如果tu，遍历加入expand tu
        *   规则2：如果appsid类型策略，遍历加入expand appsid
        *   厂商频控词典查询失败，更改为`tu`频控策略，如果tu则走规则1，appsid则走规则2
        *   不使用扩展频控

    *   设置bes_level、设置烽燧日志(记录请求耗时？）、不喜欢的cmatch列表

*   **获取rpc controller**

*   **设置partition、qid**

*   **设置usercenter_ovlexp**: 设置全流量实验参数

*   **发送request：add_request**

### handle_response

*   `get_response`：获取返回结果

*   获取rpc controller

*   记录下游ip、添加交互信息到烽燧

*   请求失败，直接return

*   解析返回`parse_usercenter_response`
    *   在curr_ctx设置usercenter_status
    
    *   如果有`freq_session`，`parse_freq_session_to_upin` => `parse_freq_session`
    
        >   *   parse_freq_session_to_upin: 将频控信息中的src_info和每个info下的多个advs添加到upin_ctx => 一次session中同一个广告位src可能显示多次不同广告？
        >   *   parse_freq_session: 将上述信息加入到curr_ctx，同时统计unitid和点击次数
    
    *   `parse_asp_show_session`，`bes_fill_show_set`或`fill_show_set`
    
        >   统计show的广告, 遍历统计过滤广告数，筛选高质量广告`_high_quality_advs`
        >
        >   *   按时间间隔: show、eshow
        >   *   按范围：match_type、planid(xbuiltin查)、ideaid(多于一次就去掉)
        >
        >   后者不过滤，直接全部show中广告放入高质量广告
    
    *   将`agg_session`放入当前检索环境curr_ctx
    
    *   离线xbox中信息存入当前检索环境`curr_ctx`
    
        *   手百物料shoubai_mt_profile
        *   长期状态`long_status`
        *   iqiyi用户标签`iqiyi_user_tag`
        *   uas信息`uas_interest`、`uas_social_relation`
    
    *   获取realtime_total_info到检索环境
    
    *   截断`bes_user_tag`
    
    *   获取`agg_info`中用户不喜欢的实体`parse_user_dislike_entity`
    
        >   遍历，获取entity_item，每个item存储多个key（用下划线连接）
        >
        >   存入当前检索环境
        >
        >   *   `entity_sign_eshow_times`：key-entity, val-eshows
        >   *   `_zhuida_state_map`： key-entity, val-允许曝光时间(last_eshow_time+time_interval)
    
*   如果没有开启意图请求`enable_uc2intentservice`，进入后处理`post_process`

    *   `set_ad_clk_seq`

    *   `fill_dedup_set_prepare`：填充去重广告

        >   记录每个物料的曝光时间(整体曝光、每个slot的曝光时间、滑动信息)

    *   预估广告停留信息actionq

    *   遍历ums用户标签

    *   `fill_dedup_set`

        >   获取厂商间隔阈值：query、winfo、planid、unitid、ideaid、hot_tag、user_id、title、brand
        >
        >   Ocpc_plus广告的阈值不同

    *   `fill_freq_control_trade_set`: 获取用户设定的要进行频控的行业

    *   `fill_eshow_trade_ratio_control_set`

## UpinProcess

### DECLEAR_MODULE_PROC_DATA

*   读取配置参数：线程数、最大历史query、intent-ann-1等

### preprare_request

*   `curr_ctx->get_upin_request`

*   `pack_request`

    *   设置search_id
    *   设置请求情况：ip、省pid、市cid、refer、original_query、flow_type(app、wap或other)
    *   根据流量来源设置字段
        *   wise：gps、代理ua、baiduid、cuid、use_uas
        *   pc: 百度id
    *   `set_cmatch_list`: 设置需要频控的广告位src，获取并设置其对应cmatch
    *   `set_show_list`

*   `write_request`

    *   申请交互的buffer和句柄

        >   因为是raw_buffer类协议

    *   add_request

### handle_response

*   获取responsebuffer

*   添加交互信息到烽燧

*   记录upin_ip

*   `read_response`

*   Pdsueserq: 设置是否需要获取pdsq预估，用于闪投信息流商品粗排

    >   目前迁移xbox

*   `parse_session_info`

    *   解析wise人群定向`parse_wise_session_info`

        >   与uas中自然属性一致，优先采用uas的

    *   历史搜索`search_session_info`

        >   term_list, adurl, asurl（最近5个）

    *   asp日志：获取shoubai_mt_profile、video_profile、nad_profile、feed_profile

        >   profile：过去72小时用户点击习惯
        >
        >   video_profile：实时播放、实时状态
        >
        >   nad_profile: 转化情况、持续时间、用户14天内点击、月转化、半年转化、用户泛化、rfm用户行为
        >
        >   feed：实时状态、feed前文广告展现
        >
        >   wise外部session
        >
        >   历史点击ideaid

    *   wise流量

        *   解析粗略位置`parse_region_info`：获取pid、cid和tag
        *   `get_locations_from_ugate`: 获取ip定位、cuid常驻点、cookie常驻点
        *   `get_gs_wise_multi_preq`: 获取wise的query（可能多阶级）
        *   `get_gs_wise_term_list`: 获取返回结果中匹配的词

*   填充upin广告去重集合

## LiteKaiwuProcessModule

### prepare_request

*   **设置请求信息**
    *   基本情况：search_id、client_id、request_type
    *   设备情况：cuid、baiduid、贴吧idfa、osid、passport_user_id、device_id（idfa/imei）

*   发送请求`add_request`

### handle_response

*   获取response和controller
*   `parse_response`
    *   获取相似人群profile和人群标签`rt_tag`
    *   遍历用户标签，记录标签及其置信度、来源和时间戳

## UmsProcessModule

### prepare_request

*   从asp中获取passport_user_id_ums，**设置uid**

*   **不同流量类型用不同字段请求**

    *   app：cuid+uid
    *   wise：baidu+uid

*   **设置请求信息**：长期兴趣元信息、图片列表、视频列表、小视频播放时间、深知用户模型、好看深知表、历史贴吧主题列表tieba_his_frs、贴吧用户首页图文点击历史tieba_news_his_index、贴吧用户首页视频点击历史、贴吧用户实时特征tieba_user_profile、tieba用户历史特征tieba_user_model

    >   long poi = 长期兴趣点（point of interest）
    >
    >   msv：小视频
    >
    >   深知用户模型：meditation_idf_score，用于获取用户attention（兴趣点及一二级分类）
    >
    >   frs: forum search, 提供贴吧按时间顺序回复的帖子列表

*   **设置登录及访问信息**：服务标签、client、token、logid、use_cache

### handle_response

>   ums会过滤三个月前的点击兴趣
>
>   [ums示例数据](http://wiki.baidu.com/pages/viewpage.action?pageId=1290911773)

*   获取response和controller

*   `parse_response`, 数据存入`curr_ctx->_ums_attr`

    *   获取上次pv的feed自然id(newsid)以及每个id的浏览时长信息

    *   解析feed浏览时长信息

    *   根据最近点击或者置信度，获取好看深知表

        >   字段：video_attention（如高尔夫）、video_category、video_sub_category、user_tech(互联网人群)、video_style

    *   如果需要获取长期兴趣点，提供高端人群的标签

        >   获取高端人群的标签，便于之后策略进行过滤敏感词

    *   解析大字版和百度app的行为

        >   [baiduapp_activity字段](http://wiki.baidu.com/pages/viewpage.action?pageId=1484678947): 周活跃、月活跃情况、feed活跃情况等

    *   根据点展统计的兴趣、获取短期兴趣、兴趣视频、一级种类、二级种类、视频种类、图片风格news_style_super、不感兴趣

    *   解析贴吧历史数据

    *   解析上次pv的浏览时长

*   如果需要获取ums意图

    *   获取intent_ctx
    *   `get_merged_intention`

## IntentServiceProcessModule

### prepare_request

*   获取request

*   `make_intentservice_resquest`

    *   获取第一个广告位信息`src_info`

    *   获取uas模块中的uas_request => `make_uas_request`

    *   填充flow_info: ip、os_id、flow_type（app、wise或pc）、cuid、baiduid、imei、idfa、oadi、android_id、Mac、user_id、caid信息（运营商等）

        >   Caid: 利用设备信息生产，替代IDFA进行归因和定向

    *   发出idm请求

        >   Idm: id merge，打通多个账号，简历账号id的映射

    *   处理厂商预取: 填充adx下发字段sid、is_fetch

        >   因为厂商的平响要求比较严格，所以需要在流量接收后到下发之前进行预取（从adx下发到feedas有十几秒延迟）
        >
        >   <img src="http://bj.bcebos.com/ibox-thumbnail98/afa6fc84a95ede7ecc4f744ec74179a1?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-03T11%3A33%3A13Z%2F1800%2F%2F0aced6ad13fa614f996174727eba78948bf9e6ccc3cec570999531fcb2fc8f01">

    *   发送请求

        *   `make_entity_request`： 针对高质流量（手百）请求意图实体。设置cuid、client_id(feedas)、tag（LONG_TERM_IT2)

            >   LONG_TERM_IT2：兴趣实体(180天内)

        *   `make_invest_will_request`：请求投资意愿，BASIC_PROFILE类型

        *   `make_behavior_request`: 请求行为数据，包括搜索query（LONG_WISE类型：180天）、点击广告（FEED_AD_CLICK: 实时24h&30天）、180天完播广告、转化广告FC_CONV、搜索query（30天）=> 设置需要对应信息的cmatch

        *   `make_ad_jianfang_behavior_request`:  曝光广告 => 设置需要jianfang的cmatch

            >   获取茧房数据：近7天最多25条曝光，对于超过总曝光数一半的meg二级行业在feedbs进行屏蔽

        *   `make_dmp_request`

            >   [dmp融合进Intent-Service设计](http://wiki.baidu.com/pages/viewpage.action?pageId=980280344)

            *    bes流量，请求ui+进行query和title改写、融合uuid和udwid
            *   封装dmp请求，设置client，需要poi、格式化地址和地点profile，计算经纬度
            *   设置应用方控制实验的flag
            *   设置基本信息：cuid、baiduid、os_id、device_id、广告位类型traffic_type、query、url、appsid、全流量实验信息和passport

        *   `uc2intentservice`

            *   获取usercenter模块信息 => `make_usercenter_request`（真实曝光数据）
            *   添加对应expose和show的cmatch到`bahavior_request`，以便获取行为数据
            *   高质流量`cmd=1`
                *   行为数据增加获取不喜欢的广告
                *   获取feed广告点击以及需要这些数据的cmatch
            *   低质流量`cmd=2`
                *   获取内容联盟的用户标签，增加需要此数据的cmatch

*   设置全流量实验参数

*   获取controller，添加请求`add_request`

### handle_response

*   获取response和controller

*   添加交互信息到烽燧

*   `parse_response`

    *   如果需要根据uas获取意图，先获取uasmodule => `parse_uas_response`

    *   `parse_uc_als_res`: 频控、广告、曝光数据

        >   用usercenter获取频控数据(freq_session)

    *   `parse_entity_response`: 获取用户意图query和广告

        >   填充`td.user_intention_ad_list`和`td.user_intention_query_list`

    *   `parse_invest_will_response`：获取投资意愿`td._invest_will`

    *   `parse_interest_response`:  获取意图兴趣

        >   `td.user_intention_interest_list`

    *   `parse_behavior_response`

        *   `parse_behavior_realtime_ad`: 获取实时广告（ideaid、userid、unitid、brand、subject广告公司名等 + 点击播放情况 => 设置频控白名单
        *   `parse_behavior_query_and_ad`: 获取30天内wise的query和转化广告列表
        *   获取用户不喜欢的广告、实体、品牌和一级行业

    *   `parse_ad_jianfang_behavior_response`: 近7天最多25条曝光，对于超过总曝光数一半的meg二级行业在feedbs进行屏蔽

    *   bes流量：

        *   `search_query_rewrite_feedbw_list` 
        *   `make_user_intent_query_bidword_item`
        *   存入`q_ctx.user_query_bidword_list`

    *   `pars_user_realtime_locations`

    *   `parse_cca_iploc`

    *   `set_return_expid`：增加data_monitor

    *   Videoq

        *   `parse_videoq_advs`
        *   `parse_videoq_emb`: 在embed_dict里面查询

    *   解析dmp

        *   `parse_ei_query_list_info`

            >   ei: entity intention, kaiwu产生的关键词扩展结构体

        *   `parse_user_basic_tag`

        *   `parse_user_locations`

        *   `parse_app_info`

        *   `parse_looklike_crowds`

        *   汇川如果有第三方流量：`parse_getui_into`

            >   个推

        *   `parse_haokan_list`

        *   `parse_pa_kt_list`: KT触发（关键词触发）的闪投

        *   增加kaiwu_query, `parse_pa_user_query`

    *   遍历获取用户，再遍历每个用户商业标签：tag_id、置信度、来源

*   如果需要获取intent_entity，获取ums_ctx，get_merged_intention

## RtaserverProcessModule

>   feedas发送两次rta预取请求，一次用于发送异步写请求，一次用于获取返回结果

### prepare_request

*   `make_rtaserver_request`
    *   `make_rtaserver_interact_info`
        *   获取支持预取的客户token list
        *   写入交互信息：请求的url、token、缓存时间
    *   按照cmd设置请求权限：只读0、只写1、写并存入redis2、异步读写3
    *   设置当次流量的src_id、cmatch、os_id、device_id、ip、media_id（只取第一个媒体行业）
*   获取rpc_controller，设置rtaserver的全流量实验参数
*   获取transport_controller，添加请求

### handle_response

*   获取response和controller
*   判断是否请求成功
*   写入rtaserver_up和返回结果，此处暂不解析，存入`q_ctx._rtaserver_res`



