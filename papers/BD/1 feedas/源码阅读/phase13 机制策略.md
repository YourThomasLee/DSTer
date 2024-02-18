

## Phase概览

*   **涉及模块**
    *   非交互类：StrategyProcessModule
*   **代码路径**
    *   `feedas/framework/`
    *   `feedas/strategy/`
*   **配置文件**：`strategy_plugin.conf`

## 执行顺序

*   **非交互类**

    基本顺序为：prepare_context => `handle_data` => debugpf

|                 | handle_data                                                  | 作用                               |
| :-------------- | :----------------------------------------------------------- | :--------------------------------- |
| StrategyProcess | 1. **运行gid级别的插件**：`traverse_global_plugins`<br>初始化plugin_manager，` run_all_plugins`<br/>① 获取gid配置以及gid级别对应的策略配置<br/>② 遍历每个子策略的插件<br/>a. 根据插件描述获取插件函数并填充插件参数<br/>b. 将ori_adv分为视频广告和非视频广告<br>c. 按照设置运行插件（两类都运行or只运行一类）:`function.func`<br/>d. 整合各形式广告列表：将pic_text_adv和video_adv放入adv_list<br/>2. **运行数据源级别插件**：`tarver_src_plugins`<br/>初始化plugin_manager，按优先级遍历广告位(高优先级先选)，遍历所有候选广告<br/>① 在当前广告位的广告队列中，删除已被选走的广告<br/>② 如果有父src_id策略，运行父src插件run_parent_src_plugins<br/>③ 否则，运行自己的插件run_src_plugins | 对召回的粗选广告队列进行过滤和精排 |

### 整体架构

<img src="http://bj.bcebos.com/ibox-thumbnail98/b899ca39108ee3bdfc2e8b80a2a63486?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-06T08%3A47%3A34Z%2F1800%2F%2F4be5926ed18588e14734bf1862da549a614fe8844433ba4cd464e135d873d6df" style="zoom: 50%;" >

*   **gid**: pv检索级别，可以查询到此次pv所处页面以及该页面有什么广告位

    *   admit[已弃用]：广告准入，过滤部分广告

        >   移入广告位级别策略的filter

    *   data_prepare

        *   异步请求观星，获取Q预估在src_id中发挥作用
        *   填充广告默认预估值和系数
        *   设置minbid
        *   广告丰富度控制

*   **src_id**：广告位级别策略，以src_id=1092(cmatch=545)为例

    *   prepare

        *   调整预估值
        *   初始化过滤阈值

    *   smart_bid

        *   对bid智能调整，CPC&oCPC
        *   计算调整后的bid和cpm

    *   filter

        *   门槛准入：根据ctr、cpm进行广告过滤
        *   依赖q值进行广告过滤

    *   budget_control：预算控制。根据广告主预算进行广告展现或过滤

        >   大部分消费逻辑在proxy中，as只有over_charge_control，针对ocpx二阶段广告

    *   dedup

        *   广告主维度去重
        *   广告丰富度维度去重

    *   price: 计费

    *   truncate：针对当前广告位各自进行截断

        >   后处理进行多数据源广告同一截断

## gid级别策略 | pv级别, 以手百gid=1为例

### admit 

*   **功能：设置准入条件，过滤广告**
*   alive：1为启用，0为不启用

|          插件函数           | 功能简介                                                     | alive |
| :-------------------------: | :----------------------------------------------------------- | :---: |
| advlist_truncate_before_asq | 对proxy和rta返回广告整体按cpm排序，过滤cpm较低的广告         |   0   |
|   blacklist_status_admit    | 按照不同维度的黑名单，过滤在黑名单中的广告<br>维度包括二级行业trade2、广告主userid和创意ideaid等 => 转入filter |   1   |
|     trade2_block_admit      | 过滤容易出敏感内容的二级行业广告                             |   0   |
|      roi_status_admit       | 统计广告主后验roi(投资回报率=利润/投入)，过滤转化低的广告<br>如果是选择加速投放(增加曝光率)，则不过滤 |   0   |
|     channel_trade_admit     | 过滤与频道相关性低的行业广告，如财经频道可能需要过滤医疗类广告 |   0   |
|       video_cpv_admit       | 过滤不能自动播放和不支持cpv计费的广告<br>只有自动播放的广告才能按cpv计费 |   0   |
|      control_cpa_admit      | 解决激励视频客户的成本问题<br>过滤已经转化的广告，降低客户成本 |   1   |

### data_prepare

*   **功能：请求观星，获取后续所需Q值**
*   业务相关概念
    *   customer center: 在线实时流，获取单元当天点击、转化、cpm扶持情况，控制扶持上线，使成功单元（达到扶持消费后退场）实时退场
    *   冷启动的退场：指广告已经成功扶持(hit)，不再需要冷启动扶持

|           插件函数           | 功能简介                                                     | alive |
| :--------------------------: | :----------------------------------------------------------- | :---: |
|      set_user_trade_new      | 遍历广告列表，设置对应的meg一级和二级行业                    |   1   |
| set_status_before_predictior | 按照各广告对q值的需要，放入不同请求队列，请求不同模型获取q值 => predictor_advlist存储<br> |   1   |
|     send_predictor_async     | 异步请求观星，与recv_predictor_async_new配套使用             |   1   |
|  send_customercenter_async   | 异步请求cutsomercenter，与recv_customercenter_async配套使用  |   1   |
|          set_status          | 等待观星返回时运行：设置各类系数(ratio、count等)以及部分之前获取的q(ctrq等)<br>关闭不必要的预算rpc |   1   |
|  recv_customercenter_async   | 获取单元当天点击、转化、cpm扶持情况，控制扶持上线，使成功单元（达到扶持消费后退场）实时退场 |   1   |
|  recv_predicator_async_new   | 接收并解析观星返回                                           |   1   |
|        set_status_new        | 观星返回后，设置相关的阈值以及q值(orig_roiq等)，包括用新q值替代旧q值等 |   1   |

## src_id级别策略

>   以src_id=1527（cmatch=908）为例，对应于 bes-app流量-激励视频adx

### prepare

*   **功能**：调整ctrq，给出广告排序分数（之后会调整）

*   **相关概念**

    *   ftype：广告投放方式 = 明投(单选+多选+优选) & 暗投

        >   <img src="https://gitee.com/WIN0624/document/raw/master/img/image-20210809163929521.png" alt="image-20210809163929521" style="zoom:50%;" />

    *   **调整ctrq的作用**

        *   **影响排序**：对CPC广告，排序依据为cpm=bid*ctrq，则提升ctrq可以降低出价bid对最终排序的影响
        *   **针对实际业务微调**：观星只能基于用户和广告本身进行预估，需要根据业务进一步调整
        *   **提高扶持流量排名**
        
    *   **ctcvrq**：只有ocpc才使用，同时用ctr和cvr获取用户向量

    *   **min_bid**：防止广告位多于广告时，最后一位广告计费为0

    *   **再营销广告**：已经注册的用户，第二题再营销，提高次日留存，实现深层转化

*   **系数说明**

    *   bid_ratio：系统出价系数，545默认为1.0

        >   bid = ori_bid ( bid_ratio)
    
*   **transfer_ratio**

    *   方式1：用transfer_q调整，根据暗投调整明投ctr

        >   <img src="http://bj.bcebos.com/ibox-thumbnail98/c343dbe8f803a12f31fa9f04c5906a78?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-09T11%3A34%3A42Z%2F1800%2F%2F9e6dbf694c6e0f9b8f907d4cf035618be6d0d049b004347026aa4ef9110c4cab" style="zoom:50%;" >

    *   **方式2：按比例调整**

        ctr = ori_ctr * xxx_ratio()

    *   **方式3：线性调整**

        ctrq = ctrq*w + b

*   **计算广告分(CPM)**

    *   Ue-loss：用户体验损失，根据clkq和ctrq算出，ctrq越大，用户体验越好，损失越小

>   CPM广告，bid直接决定score，因为bid就是按千次进行付费
>
>   单位变化：元变为分，factor=10^6
>
>   <img src="http://bj.bcebos.com/ibox-thumbnail98/9ff6331f97501adb60de8a34cb2e7647?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-09T11%3A51%3A37Z%2F1800%2F%2Fa71bf0712db1e9f0da00712c092e08ebf44cb4edcc76ca60f18a599cd1cb89e6" style="zoom:50%;" >
>
>   <img src="http://bj.bcebos.com/ibox-thumbnail98/f9943bfe204410c135ab4cd0643ea544?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-09T12%3A06%3A11Z%2F1800%2F%2Fe1db62409efdecdfed3cc4224f4498a94a7c58af5b3cdac016b6e530d5fb66e7" style="zoom:50%;" >
>
>   *   非cpm广告：按点击CPC或其他行为付费，需要借助CTR转化为CPM
>
>       t变换强调ctrq的重要性（ctrq和bid共同决定score），用t控制广告质量&bid对最终排序的影响程度，t越大说明ctrq越重要 => 原本bid只代表一次动作的出价，现在需要转换为一次展示的出价，pricesort_q理解为转换系数
>
>       *   配置读取q_t
>
>       *   T变换：此处ctrq已经乘了factor10^6，而单位变化只需要x1000（千次）x100（元变分），所以还需要除10，才为10^5
>
>       *   计算分数：q*bid
>
>       *   用分数计算ctr和clk，进而计算ue_loss
>
>       *   用计算出的ue_loss得到最终得分
>
>           <img src="http://bj.bcebos.com/ibox-thumbnail98/c2b73093c1a67fabb19b51eb1e3a02db?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-09T12%3A13%3A36Z%2F1800%2F%2F3dd6093e593b90140c66df6a1e86433b477bd945d480564e716a8f1baf019dfb">

|         插件函数         | 功能简介                                                     | alive |
| :----------------------: | :----------------------------------------------------------- | :---: |
|    set_status_mul_src    | ① 设置明投标志位`is_mingshi_src_id`<br>② 设置每个广告的出价阈值`min_bid`<br>③ 根据广告投放类型，设置bid_ratio（0为优选，1为单选） |   1   |
| set_ocpc_to_ocpm_mul_src | 根据本地词典，将命中词典的二阶段ocpc广告，随机替换为ocpm     |   1   |
|    select_idea_srcid     | 过滤idea对应src_id与当前广告位不一致的广告`adv->idea_src_id != adv->src_id` |   1   |
|      transfer_ratio      | ① 按照不同维度，调整ctrq以影响广告排序<br>a. 人工调权<br>b. 按比率调整: 完播率eplay、组合样式、游戏扶持、程序化创意、新用户&游戏客户&详情页相关扶持取最大ratio、广告主类别、视频广告客户粒度、视频广告（分行业&不分行业）<br>c. 线性调整: 样式<br>② 给出排序分数score（后面会随着bid改变而改变）<br>CPM和非CPM<br>③ 降低最近已刷ideaid的ctrq，取倒数；如果过低，则重置 |   1   |
|   bes_adjust_for_sort    | 在按价格排序之前，借助`support_ratio`调整价格q`pricesort_q`，提升某些广告的排序得分score（冷启动、rta等） |   0   |
| set_bes_support_cpmdelt  | 对cpc广告或者deeplink等广告，设置`adv->bes_cpm_delta`和`bes_support_type`，delta_cpm之后有助于提高整体排序`cpm=ori_cpm+delta_cpm` |   0   |
|  set_bes_strong_support  | 强扶持插件：再营销人群&游戏人群，查询词表获取扶持系数，更新`ad_cpm_ratio`和`bes_support_adcpm_ratio` |   1   |
|   set_bes_cpm_support    | 弱扶持插件：<br>① 强扶持、冷启动、tu黑名单、已退场or达到退场门槛的广告不走该逻辑<br>② boost爆量扶持：三种扶持系数<br>a. 排序扶持: delta_cpm = (a-1) * x + b, a和b可累乘&叠加<br/>b. 计费打折：bes_support_price_ratio累乘temp<br/>c. 报价扶持：bes_support_adcpm_ratio累乘temp<br>③ pv刷次扶持: 累乘三种系数<br>④ 智能人群弱扶持：累乘三种系数 |   1   |
|  bes_coldboot_adv_quit   | 冷启动退出扶持：<br/>① 满足整体预算，则退场<br>② 对冷启动广告，获取cpm、click、cons阈值，判断三者任一是否达到退场条件<br>③ 白名单中的广告不实施退场逻辑，其余查阈值，判断历史累计点击/转化数是否超阈值 或 使用当天数据进行判断 |   1   |

>   累乘&叠加
>
>   <img src="http://bj.bcebos.com/ibox-thumbnail98/269cfaf22d5963ea9045cf704c200477?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-09T12%3A57%3A11Z%2F1800%2F%2F7fd8eb106f8a5b646d3e8b48d9ed63b726f7e2132bc11ce6c8ab2d680f3093ba">

### smart_bid

*   **功能**：对非CPM广告调整出价，重新计算排序得分cpm

    >   ocpc一阶段就是cpc出价方式

*   **业务概念**

    *   激活类型transtype

        >   <img src="http://bj.bcebos.com/ibox-thumbnail98/72d29cd319f51d9ea693f31f1b8fa963?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-09T15%3A07%3A23Z%2F1800%2F%2Fdd6721b14502e5dfaa85b5f1d09be966ca7b29ec6b367bb51050367d1d94726b" style="zoom: 50%;" >

    *   Match_type: 触发分支 & mining_type: 触发类型

    *   **调整系数**

        *   **作用**：ocpc二阶段使用的roiq可能不准，需要调节

            >   ocpc_bid = bid * roiq => 最终还是要按点击计价，所以将转化出价bid变为点击出价

        *   **反馈系数**：ocpc_bid/cpa，表示广告主转化出价和系统转化出价的误差

            *   cpa是转化成本 = charge/conv，对应百度每次转化要求的收入

                >   点击总收入/转化数，这个数字高，说明很多点击才够一定转化，roi高

            *   ocpc_bid是广告主对每次转化的出价

            *   reach_adjust_coe < 1，真实转化成本高于出价，广告主多付钱（因为点击多），需要调低roiq，降低每次点击的付费

            *   reach_adjust_coe = 1，成本恰等于出价，roiq准确

            *   reach_adjust_coe > 1，真实转化成本低于出价，广告主少给钱，说明cpa少，少量点击就可以达成一定转化，需要提高每次点击的成本，调高roiq

        *   **计费比倒数**：ocpc_bid/price，一般设为1.0

    *   **有效播放归因**

        *   **提出背景**：对于视频即使没有点击，只要完成有效播放都有助于转化

    *   **客户回传数据**

        *   feedroiq：广告主回传各维度转化数据roiq
        *   Feedpayq: 广告主回传付费金额

|      插件函数      | 功能简介                                                     | alive |
| :----------------: | :----------------------------------------------------------- | :---: |
|   user_smart_bid   | CPC溢价策略，调整CPC和OCPC一阶段广告出价，重新计算调整后的bid和排序得分.<br>① 跳过不使用bid_ratio的CPM、OCPC第二阶段和明示单选的广告，单选广告设bid_ratio=1<br>② 调节bid_ratio：src_id、os、mt维度、rta溢价策略(`rta_bid_rise`作为溢价系数)<br>③ 重新计算pricesort分数：bid*bid_ratio代替原分数计算的bid |   1   |
|  anti_virtual_bid  | CHECK_STAT，不允许空广告和空出价                             |   1   |
|      ocpc_bid      | OCPC溢价策略，调整ocpc二阶段广告的出价，重新计算cpm<br>① roiq = bid_ctcvrq*Q_FACTOR / pricesort_q<br>② 调节ocpc_bid_ratio<br>a. 按不同维度初始化：转化类型transtype、os、ann触发<br/>b. 累乘：virtual_mt虚拟样式、基础样式、组件样式、程序化创意、rta溢价系数等<br>③ 游戏扶持和游戏用户，使用反馈系数`reach_adjust_coe`：<br>a. 游戏扶持：unit或user维度ratio<br>b. 游戏用户：当前cmatch的ratio<br>④ 设置两个调整系数`price`和`reach_adjust_coe`<br>a. 使用billing_ratio: bid = ocpc_bid * roiq * billing_ratio<br>b. 不使用billing_ratio：bid = ocpc_bid * roiq * reach_adjust_coe * price_adjust_coe * ocpx_bid_ratio<br>⑤ 计算排序得分pricesort_score<br>a. 二阶段oCPC(按转化出价,变成按cpm)：ocpc_bid * ctcvrq / 10^7 * reach_adjust_coe * price_adjust_coe * final_ocpx_bid_ratio<br>b. 非CPM(按点击变成按cpm)：pricesort_q * bid |   1   |
|   bes_smart_bid    | 品牌维度调整出价<br>① ocpc二阶段广告跳过此逻辑<br>② 获取cmatch和tu对应的bes_bid_ratio => 据此获取brand_bid_ratio<br>③ 调整出价bid = bid * brand_bid_ratio * bes_bid_ratio<br>④ 用新的bid计算排序得分和multarget排序得分：也是pricesort_q * bid (-ue_loss) |   1   |
|  calc_eplay_score  | 对ocpc二阶段广告进行有效播放归因<br>① 过滤不使用有效播放归因的单元unit和广告主userid<br>② 设置最长和最短播放时间<br>③ 目前仅支持物料维度，设置有效播放门槛eplay_check_time<br>a. 当广告播放时长大于[max,视频原时长-1.5秒]中的任何一个时间记作一次有效播放 => 取min<br>b.同时使用min值作为兜底，防止出现原时长-1.5秒极其小 被误认为有效播放的情况 => 取max<br>④ 计算有效播放的增益：<br>a. 所需系数：eplay_cpm_ratio；不调整ceplayq<br>b. 计算cpm：ocpc_bid * (eplay_conq/Q_Factor) * eplay_pricesort_q (eplay/show) * adv->reach_adjust_coe * adv->price_adjust_coe * adv->final_ocpx_bid_ratio * eplay_cpm_ratio<br>c. 最终cpm=原cpm + 有效播放cpm |   0   |
| ocpc_deep_obid_pk  | 深度转化成本优化，智能调节双出价<br>① 跳过OCPC二阶段广告<br>② 获取转化率<br/>a. shallow_roiq = adv->roiq / Q_FACTOR<br/>b. deep_roiq = adv->deeproiq / Q_FACTOR<br/>③ 计算双出价<br/>a. 浅层转化出价shallow_bid = ocpc_bid * shallow_roiq；bid_ratio = bid/shallow_bid<br>b. 深层转化出价deep_bid = deep_ocpc_bid * deep_roiq * shallow_roiq * bid_ratio<br>④ 计算最终出价和得分<br>a. 获取深度转化权重`ocpc_deep_bid_weight` => 计算浅层权重`ocpc_deep_shallow_weight`=1-ocpc_deep_bid_weight<br/>b. Adv->bid =shallow_bid * ocpc_deep_shallow_weight + deep_bid * ocpc_deep_bid_weight |   1   |
| feedpayq_bid_ratio | 暗投根据预估档位调节bid<br>① feedpayq_bid_ratio = 本单元预估feedpayq / 历史7天平均feedpayq<br>② bid = bid * feedpayq_bid_ratio<br>③ score = pricesort_q * bid => multarget直接-ue_loss |   1   |

>   ```c++
>   adv->eplay_check_time = std::max(min_eplay_duration_threshold,
>                                    std::min(max_eplay_duration_threshold, float(ori_duration) - MIN_VIDEO_DURATION_BUFFER));
>   ```

### filter

*   功能：根据ctr、cpm进行广告过滤

    >   大部分消费逻辑在proxy中，as只有over_charge_control，针对ocpx二阶段广告

|        插件函数         | 功能简介                                                     |    alive    |
| :---------------------: | :----------------------------------------------------------- | :---------: |
| blacklist_status_filter | 黑名单过滤，维度包括：二级行业、实体、广告主、计划、winfo、作弊、疑似作弊等 |      1      |
|       thr_filter        | 多种阈值过滤<br>① 根据cuid获取cpm和ctr门槛<br/>② bes-boost、bes-冷启动、特定触发方式广告跳过过滤插件<br/>③ 分触发方式调节cpm和ctr门槛，作为base门槛，获取门槛ratio，更新门槛<br>④ 加速投放的广告通过delta降低门槛：acc_ctr_thr = base_ctr_thr - delta，计费时代价是乘计费比<br>⑤ 分地域和行业调节cpm和ctr门槛<br>...<br>⑥ 排序所有门槛，取最大的base+delta作为门槛 => 游戏客户门槛扶持、冷启动调低门槛<br>⑦ 根据门槛截断 | 1 1242-2904 |
|        mt_filter        | ① 根据show_control_exp.conf配置，进行基础物料样式过滤<br>② 对特定的src和mt做物料校验，以支持竖版视频横版场景播放：二期暗投 \|\| 原明投 \|\| 原暗投，都不是则过滤<br>③ 过滤非法程序化竖版视频<br>④ 明投和程序化创意不过滤 |      1      |
|     user_roi_filter     | 根据user roiq阈值进行过滤，转化率低的广告主广告直接过滤      |      1      |
|     deeproiq_filter     | 根据deep roiq阈值进行过滤: <br>① 获取深度转化门槛值：unit > plan > user<br>② deep roiq小于门槛值的广告被过滤 |      1      |
|   control_cpa_filter    | 激励视频特有，相同设备相同流量的已转化用户不重复返回相同广告 |      1      |
|       cvr_filter        | cvr阈值过滤.用app_sid和cmatch获取cvr门槛，cvr_roiq低于门槛的广告被过滤 |      1      |
|    bes_device_filter    | 过滤和设备不匹配的广告                                       |      1      |
|  k12_whiltelist_filter  | 针对k12转化人群，需要根据其srcid和unitid白名单进行过滤：不再任一白名单的广告被过滤 |      1      |
| feedroiq_credit_filter  | 过滤授信转化类型的ocpc二阶段广告                             |      1      |
| bes_newstyle_mt_filter  | 联盟组件样式白名单准入                                       |      1      |

### budget_control

*   **功能**：针对匀速消费的广告进行展现控制 => 以小时为粒度，若当前小时内消费超过预计消费阈值，则停止展现

*   **业务概念**

    *   三种投放

        *   标准：正常投放，bid > 余额，广告被过滤
        *   匀速：若某段时间投放过多，则停止，过一段时间再投放
        *   加速：降低ctrq，加速广告投放，price提高

    *   **消费空间控制**

        ```c++
        // 剩余预算比例：剩余预算 / 总预算
        left_budget_ratio = (adv->plan_feed_budget - adv->plan_total_consume) * 1.0 / adv->plan_feed_budget;
        // 剩余消费空间，如果之前消费过多
        left_consume_space_ratio = (all_day_ratio_sum - passed_hour_ratio_sum * buffer) / all_day_ratio_sum;
        ```

    *   **消费速度控制**
        *   ocpc二阶段：constant_speed_delivery_with_pacing
            *   计划投放时间的前n分钟内限制投放概率
            *   超出小时级别消费阈值，停止展现
            *   基于pacing策略
        *   非ocpx
            *   前两项同上
            *   使用分钟粒度控制

|      插件函数       | 功能简介                                                     | alive |
| :-----------------: | :----------------------------------------------------------- | :---: |
| over_charge_control | ① **execute_budget_pacing_v1**<br>a. 只对ocpx二阶段和字典中存在的plan生效<br/>b. 获取广告投放计划：周几、什么时间<br/>c. 计算所有可投放时间的预计消费比率all_day_ratio_sum、计算已过去可投放时间预计消费与当前时间段预计消费比例<br/>d. 过滤之前投放太快和预算不足的<br/>e. 获取roiq动态门槛，过滤roiq不足的<br>② **constant_speed_delivery**<br>③ 遍历广告，实现bt_filter（预算门槛）和vc_filter（虚拟计费）两个维度过滤 |   1   |

### dedup

*   **功能**：去重

|  插件函数  | 功能简介                                                     | alive |
| :--------: | :----------------------------------------------------------- | :---: |
| dedup_proc | cpm(pricesort_score)降序排序，按user_id、plan_id、unit_id维度去重 => 实质是保留各个维度最高cpm的广告<br>① 监控过滤后剩下广告数、视频冷启动广告扶持(修改排序得分)、bes冷启动跳过去重逻辑<br/>② 按照multarget权重或cpm或multarget cpm排序<br>③ 遍历广告，去重维度：主体、品牌、基础物料、标题、二级行业、实体、视频 |   0   |

### price

*   **功能**：给出广告最终计费
*   **业务概念**：三种竞价方式

|           插件函数            | 功能简介                                                     | alive |
| :---------------------------: | :----------------------------------------------------------- | :---: |
|        calc_vcg_price         | 计算vcg收费                                                  |   0   |
|         calc_gsp_bes          | 计算广义第二价格收费<br>① 按cpm排序，设置截断数量(截断数和请求数的最小值)<br/>② user_id和planid层级去重<br/>③ 如果有下一个广告，gsp_score=next_adv->multarget_pricesort_score<br/>a. CPM广告或ocpc二阶段，只用单位变化，gsp_price = gsp_score / 100 + 0.5<br/>b. 非CPM，变回cpc价格，gsp_price = gsp_score / pricesort_q + 0.5<br>④ 没有下一个广告<br/>a. cpm&ocpc二阶段：Gsp = min_cpm / 100 + 0.5<br>b. 非cpm：gsp = min(minbid, calc_price_cpm/pricesort_q + 0.5) (bes_gsp_update); 非update，gsp = minbid<br>⑤ 调节费用<br>a. 第三方adx请求广告，需要ratio调节bid、price<br>b. 阈值处理：不能低于minbid |   1   |
|       calc_first_price        | 用分数反求出价<br>price = (multarget + ue_loss) / pricesort_q / first_price_ratio + 0.5 |   0   |
|       calc_mincpm_price       | 防止流量贱卖<br>① 对ocpc第二阶段、非CPM和CPM广告，各自设立mincpm_price<br>② 基于mincpm调整广告价格，低于此的要调高<br>③ 调整bid，禁止溢价的广告，需要调回ori_bid |   1   |
|       bes_adjust_price        | 联盟流量调价，OCPC广告跳过此逻辑，其余广告price=price*adv_price_ratio |   1   |
| bes_price_upper_lower_limits  | 联盟流量明投扶持，明投设置价格上下限<br>① OCPC二阶段价格不能高于原本出价<br>② 价格控制下限min_bid，上限max_price |   1   |
| bes_coldboot_cpmdelta_support | 联盟流量冷启动cpm_delta扶持（排序扶持）<br>① 再营销广告直接返回<br>② 扶持词典中的单元，修正mindBid和mincpm为冷启动设定值<br>③ 按cpm排序，遍历<br>a. 获取冷启动广告的扶持系数bes_cpmdelta_support_ratio<br>b. 计算转化数：cur_conv_ratio=unit_conv/（convs_thr+0.1)，转化数高则降低扶持系数，反之调高<br>④ 跳过强扶持(1)和超纲扶持(cpm+cpm_delta*support_ratio > top1_cpm)<br>⑤ 找到第一个冷启动广告（非第一位）<br>a. 计算冷启动广告的ad_cpm(用第一位广告的price计算）: CPM类型和CPC类型<br>b. 修改排序得分，将冷启动广告放到第一位<br>c. 计算冷启动扶持损失cpm<br>⑥ 重新排序 |   0   |
|   bes_all_cpmdelta_support    | 对所有需要扶持的广告进行上述扶持，不断调整至第一位、修改cpm、重新排序 |   0   |
|        bes_all_support        | bes整体扶持插件                                              |   1   |

>   1.  **判断当前流量类型**
>
>       *   不是强弱扶持或冷启动跳过此逻辑
>       *   修正bid：强扶持广告修改minbid，重算得分
>       *   修正price：price很小广告，用mincpm计算。price = min / 100 + 0.5
>
>   2.  **筛选首位广告**
>
>       * 计算adv_top1报价
>
>           * 非cpm：ad_cpm_ratio * price * ctrq 或 调整好price后，price*ctrq
>           * Cpm: ad_cpm_ratio * price / 100 或 price / 100
>
>       * 记录adv_top1原始指标
>
>       * 如果第一位已经是扶持广告，需要做修正
>
>           * 更新弱扶持广告的扶持cpm
>
>               <img src="http://bj.bcebos.com/ibox-thumbnail98/9367d38cc53de4d8ee7fd9bbe569a1f3?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2021-08-10T08%3A01%3A35Z%2F1800%2F%2F70e83cf0161aa025395df5307bb5b952f752c25690bad50bf8c5ad12f5cd5717">
>
>           * 寻找底价(第一报价)，若高于二价，则修正报价扶持系数
>
>           * 累乘计费打折系数，并修改最终系数
>
>               `ad_cpm_ratio`=bes_support_top1_ratio1* top1_ratio2 * support_adcpm_ratio
>
>           * 更新报价，直接返回
>
>   3.  **筛选扶持广告**
>
>       *   强扶持 > 弱扶持-其他 > 弱扶持-冷启动
>       *   强扶持：选ctr最大的或cpm最大的
>       *   弱扶持-其他：随机筛选或最大cpm
>       *   弱扶持-冷启动：扶持后cpm大于top1【multarget + (a-1)*multarget + support_cpm_b】
>
>   4.  **确定扶持广告，重新计算ad_cpm**
>
>   5.  **调整扶持广告**
>
>       *   计算price：用gsp_score，或mincpm
>
>       *   打平报价差：将当前扶持广告报价通过ratio1回调
>
>           >   ratio1 = 原top1 cpm / (ad_cpm_ratio * price * ctrq)
>
>       *   底价大于二价需要修正
>
>       *   更新报价系数：ad_cpm_ratio = 三个ratio连乘
>
>       *   更新报价：CPM和非CPM
>
>   6.  **统计损失cpm = 原始1位cpm - 扶持广告cpm**

### truncate

*   **功能**：将广告队列截断至固定长度

|       插件函数       | 功能简介                                                     | alive |
| :------------------: | :----------------------------------------------------------- | :---: |
| default_show_control | src_id级别广告截断，单数据源广告位show_control  ① 按cpm排序，进行广告截断  ② 统计剩余广告类型及数量 |   1   |
|   post_set_status    | 设置ocpc_level、ori_bid等信息  设置bid_type、ocpc_level、ori_bid等信息 |   1   |



## reference

*   [feedas检索流程策略](http://wiki.baidu.com/pages/viewpage.action?pageId=710756218)
*   [策略插件学习](http://wiki.baidu.com/pages/viewpage.action?pageId=1568500048)
*   [feedas机制策略阶段详解](http://wiki.baidu.com/pages/viewpage.action?pageId=1320242359)
*   [strategy](http://wiki.baidu.com/display/~yangzhifeng01/strategy)

