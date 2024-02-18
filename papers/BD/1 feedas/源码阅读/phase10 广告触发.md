## Phase概览

*   **涉及模块**
    *   交互类：FeedProxyProcessModule
    *   交互类：RtaBsProcessModule

*   **代码路径**：`feedas/framework/`

## 执行顺序

*   **交互类前期**

    *   每个非交互类的执行顺序`create_rpc_channel` => `prepare_request` => `send_request`

    |      Module      | prepare_request                                              | handle_response                                              | 作用                                                         |
    | :--------------: | :----------------------------------------------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
    | FeedProxyProcess | 主要函数：`make_feedproxy_request`<br>1. 打包common_info:  ① 当次请求的信息(src_id、pid、cid)<br> ② 设备信息(cuid、baiduid、passportid、os_id、device_brand_id) <br/>③ 用户信息(gender、age、user_locations、人生阶段、婚姻、app信息、dt_app、dislike、dislike频控、茧房) <br/>④ 透传&下发数据：general_trans(caid、bes文章id&标题&作者&种类等)、查询mtq、deeproiq所需的userq&xbox_name等、海外流量广告标记、百青藤流量下发rta预取结果、实时广告列表<br/>⑤ 检索关键词：词包id 等 <br/>⑥ afd上游信息(epvq曝光率模型q) <br/>⑦ 过滤信息：物料白名单 + bes定义的媒体保护(不想出的广告、行业等) + 视频长度&商业价值 <br/>⑧ GD下游需要的信息：app_name、os_version、ip、屏幕长宽、ua等 <br/>⑨ 智能人群召回信息：设置人群tag和优先级、media_user_id<br>2. 打包upin_info：历史创意、品牌、行业、广告主、主题、人群定向app、历史曝光次数的统计table<br>3. 打包freq_control_info：upin_dedup_set(winfo、user、ideaid、planid、unitid、title、实体的点击和曝光时间)；茧房频控；shown_info频控；深度转化ocpc_deep频控<br>4. 发送bs请求：将数据存入`feed_info`、`geed_info`和`common_info`<br>① feed_info: 流量类型、教育水平、feed&好看&upin人群定向兴趣、金门的意图触发关键词、设置huichuan改写query、意向人群、相似人群包、original_query(8小时和7天、ei_query、贴吧名、爱奇艺视频标签都算)、文章标题、广告标题、完播广告列表、设置用户向量feedannq&roiquser ...<br>② geed_info: 新兴趣<br>5. 发送闪投请求：<br/>① 添加kaiwu信息：rt人群标签、kt关键词、相似人群<br/>② 设置pamixer_info：人群定向包、意图关键词、流量来源、下发user特征<br/>6. 发送gd请求，设置geed_info：流量类型、地点、兴趣、用户标签检索词、不喜欢的广告、人群信息、实体query、挑选兴趣和权重以便之后匹配GD<br/>7. 添加缓存信息 | 主要函数：`parse_feedproxy_response`<br/>① 解析投放信息：广告主选定投放类型(标准、匀速、加速)、投放版位信息<br>② 解析广告信息：广告主id、行业id、意图信息-rsq(query级别ectr)&pvr(有效检索比率epv/pv)、兴趣标签、广告下载数、评论数、评分、标题签名、品牌签名、深度转化类型、转化率和ocpc出价；pamixer独有字段<br>③ 解析物料信息: 长宽、视频长度、视频文件大小、程序化创意(之后进行优选）<br>④ 解析作弊信息: 疑似作弊层级、作弊原因、惩罚标识<br>⑤ 其余信息：冷启动支持、风险系数 | 并行访问bs、闪投、GD，进行广告触发，返回广告召回初步队列     |
    |   RtaBsProcess   | 主要函数：make_rta_request<br>① 填充user_info：age、gender、pid、ip、cid、edu、location、app信息、相似人群、otpa人群、mt白名单、频控信息、婚姻、人生阶段<br>② 填充intent_trigger_info：媒体信息<br>③ 填充video_info | 主要函数：parse_rta_response<br>① 解析广告信息：分时投放、投放类型、投放版位、作弊情况、风险系数、程序化创意信息<br>② 如果确认投放，打包广告内容到original_advlist | ① 用rtaserver预取结果进行前置过滤<br>② 只读模式请求rtaserver，询问外部广告主是否参与该广告位的实时竞价<br>③ 获取RTA物料（单独存储） |

    >   *   feedproxy
    >       *   前卡广告：针对泛娱乐中间页的引流广告，即点击后倒流到泛娱乐中间页上,且点击该广告不直接对广告主进行计费
    >       *   智能人群：探索定向之外的人群，在控制成本的情况下获取更多转化的定向优化产品
    >       *   作弊信息：媒体作弊 or 广告作弊，提高广告质量
    >   *   rtabs
    >       *   LT意图人群：LBS定向，地理位置和商圈
    >       *   otpa人群：广告主定向人群

*   **交互类后期**

    *   基本顺序为：wait_response => handle_response => close_rpc_channel

