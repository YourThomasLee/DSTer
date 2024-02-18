## Phase概览

*   **涉及模块**
    *   交互类：AdrestProcessModule
    *   非交互类：FeedAdrestXboxModule
    *   交互类：MaterialProcess

*   **代码路径**：`feedas/framework/`

## 执行顺序

*   **交互类前期**

    每个非交互类的执行顺序`create_rpc_channel` => `prepare_request` => `send_request`

|     Module      | prepare_request                                              | handle_response                                              |                             作用                             |
| :-------------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------: |
|  AdrestProcess  | 主要函数：make_adrest_request<br>① 填充adrest_request: 广告信息(idea、winfo、unit、plan等)<br>② 填充ad_req_share: qid、src_id | ① 获取广告位的样式名单：get_src_mt_set<br>② `parse_adrest_response`：获取高级样式、新样式(统计出现次数作为频控信息)、动态标题；渲染闪投程序化创意；闪投和反向流量继续获取程序化创意；下发样式过滤黑名单<br>③ 程序化创意(下移bs)：获取两阶段物料；选择一阶段或二阶段 => 一阶段随机选一组物料作为当前广告物料、二阶段优选(mtq)<br>④ 其余物料处理：动态标题、图像增强 | 与Adrest模块交互，获取程序化广告物料，与建筑大师元素拼接组合(二阶)，创意优选 |
| MaterialProcess | 主要函数：`make_material_request`<br>① 填充uas_log：年龄、性别、工作等uas信息；兴趣信息；定向人群的app信息<br>② 填充res_share: pid、cid、cuid、地点、广告信息(user_id、winfo、plan、unit、title)等 | 解析target_url，存入`_mserver_api_response`                  |           请求落地页服务，获取每个物料的target_url           |

>   *   广告物料的三种url链接
>       *   show_url：宏替换后的广告主原始落地页裸连url
>       *   target_url：计费串中包裹的链接，计费链接跳转的链接
>       *   mt_newstyle中url：广告主物料中未被宏替换的url
>   *   程序化创意
>       *   作用：将广告主从针对不同广告位提供创意的方式中解脱出来，实现广告对不同流量的智能适配
>       *   第一阶段：广告组合处于探索阶段
>       *   第二阶段：广告组合通过投放效果验证处于优选状态
>       *   退场：广告组合投放效果不佳，不再继续投放
>       *   离线q：没有用户特征情况下对组合进行质量预取
>       *   分批探索：根据离线q排序后取一阶段组合top n和二阶段组合合并，作为分批探索的候选集
>       *   mtq：预估在线广告质量的q
>   *   MTQ下移
>       *   因为原本bs使用bsq对默认组合和普通创意排序，as才用MTQ进行组合优选
>       *   但组合多且组合差异大，导致BSQ对程序化创意广告预估不能提现组合信息 => 下移MTQ，升级bs的广告排序策略

*   **非交互类**

    基本顺序为：prepare_context => `handle_data` => debugpf

|                | handle_data                                                  | 作用                               |
| :------------- | :----------------------------------------------------------- | :--------------------------------- |
| FeedAdrestXbox | 1. program_new_xbox: 新框架下不请求<br>2. architecture_xbox<br>① fill_request: key为ideaid和xbox_version；记录广告请求建筑大师xbox的次数<br>② parse_response：解析建筑大师，存入`adv.arch_master_vec`；记录建筑大师返回成功数量 | 查询建筑大师词表，获取建筑大师元素 |

>   *   建筑大师
>       *   作用：通过内容识别技术对客户提交的元素物料进行二次加工，智能适配流量
>       *   注意：视频详情页(cmatch=585)仅支持小图样式
>           *   小图为3张，为客户同时生成三图投放
>           *   小图>=4，小图推送检索端，检索端选出散图

*   **交互类后期**

    基本顺序为：wait_response => handle_response => close_rpc_channel
