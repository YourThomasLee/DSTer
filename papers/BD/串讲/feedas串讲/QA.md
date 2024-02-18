1. 广告有哪些物料样式，mt样式物料包括哪些？过滤是怎么样的？

名词解释：
物料：广义上讲为各种形式的材料，在检索系统里对应的是xbox、xbuiltin数据，以及adv广告数据
样式：广告展现结果，主要以json形式描述；
mt(model type, 广告创意样式编号)：记录了一种广告需要的各种元信息，如大图，单品三图，三品三图，单品单图，小图链接，三图链接，大图链接，小图下载，三图下载，大图下载，小图视频，大图视频，三图电话，大图电话，橱窗，大图视频下载，竖版视频链接，竖版视频下载等等；具体可以查看[线上样式](http://mt.baidu.com/materials/mt)

mt过滤机制写在feedbs模块, 包括白名单过滤，黑名单过滤，图片尺寸过滤，视频播放时长过滤

| 策略         | 说明                                                         | 备注                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 白名单过滤   | 上游下发白名单+[src_mt_exp.conf](http://icode.baidu.com/repos/baidu/ecom-release/feedads-prod/blob/master:feedbs/conf/src_mt_exp.conf)配置的样式(数据源和样式对应关系)，不在配置中的样式过滤 | 业务端配置有是否允许降级开关，如果该开关打开，并且满足了上述黑白名单过滤条件，则滤掉组件样式，保留基础样式。白名单：AFD计算得出，检索端配置的支持样式是全集，然后针对每次pv，afd会计算一个支持的样式，然后检索端会进行白名单过滤。（AOP也有白名单，目前还没有实现） |
| 黑名单过滤   | 上游下发的黑名单过滤                                         |                                                              |
| 图片尺寸过虑 | 1、如果请求中不带尺寸信息，则过滤idea层级中不含图片的物料 2、如果请求中带有尺寸信息，则如果请求的宽高比，不等于idea的图片的宽高比，则过滤idea 3、图片物料尺寸如果不满足请求的尺寸比例，则过滤; |                                                              |
| 视频时长过滤 | 属于视频物料，并且视频时长大于请求中设置的最大视频时长则过滤 |                                                              |

参考：[原生样式相关业务](http://wiki.baidu.com/pages/viewpage.action?pageId=697086706),    [凤巢广告使用了哪些mt物料信息？分别代表了说明含义？](http://wiki.baidu.com/pages/viewpage.action?pageId=730422782),   [mt样式](http://wiki.baidu.com/pages/viewpage.action?pageId=798796374) , [广告样式MT梳理](http://wiki.baidu.com/pages/viewpage.action?pageId=1155258647), [mt样式处理](http://wiki.baidu.com/pages/viewpage.action?pageId=540643753)

2. 频控是什么概念？用什么信息做频控，有哪些维度的频控？

频控主要是针对一段时间内的pv而言的，在某一段时间内，控制某个广告对于某个用户的展示次数，优化ctr、cvr，防止ctr、cvr的大幅度下降 (如果一段时间内向某用户反复展示一个广告，而得不到实际点击和转化，这会使得ctr、cvr大大降低)
频控的作用：1. 提升用户体验;  2. 提升整体的点击转化率;  3. 广告主预算有限情况下增加更多受众;
频控主要依据两个信息来做频控：

feedas的频控粒度分为pv频控、query频控、广告频控。频控策略生效位置：非程序化广告在bs和adplus，程序化广告在请求adrest之后做title、pics元素级别的频控，闪投（非白名单广告）则在feedproxy模块。频控的信息依据主要来自于两块：

- afd的show_info频控(存储用户近8小时的20条广告)，数据由afd 存储在redis，访问检索端是传入feedas，在feedbs做频控过滤，实时生效的频控策略，最近20条广告频控。show_info频控又分为三种：

  - show_win：存储竞胜的广告。在广告返回时写入redis。 列表页，需配置
  - eshow：存储曝光的广告。在广告曝光时写入reids，延迟10ms以内。默认配置。
  - eshow+last_showwin： 存储曝光的广告和最后一次竞价胜出的广告，用来解决相邻pv滑动过快导致的频控失效，仅545，719

  showinfo策略：每个广告位配置三种频控策略中的一种，在afd的广告位配置文件中可[查询](http://aop.baidu.com/ssp/adplace/placeList)

- 检索端的user_center频控。通过实时流将用户曝光数据存入redis，在feedas获取，在feedbs进行过滤。由于实时流延迟，通常会有1s至几分钟的延迟。目前UserCenter支持的频控维度：

  | User            | 广告主                     |
  | --------------- | -------------------------- |
  | planid          | 计划                       |
  | title           | 标题                       |
  | winfo           | 广告主选词                 |
  | ideaid          | 创意，query pv请求级别内容 |
  | brand           | 品牌                       |
  | match_type      | 触发类型                   |
  | triggered_query | 触发query                  |
  | trade           | 行业                       |
  | subject         | 实体，广告公司的名称       |

usercenter收集的频控信息和 shown_info中的频控信息主要有两点区别：1）数据来源不同，usercenter的数据来源是历史曝光数据，而shown_info是feedas上游传下来的最近10次pv的检索数据。2）时效性不同，usercenter支持的是长时间（最长8小时）的频控，而shown_info中的频控信息由于是最新10次pv的检索数据，一般就几分钟

参考：[show_info频控](http://wiki.baidu.com/pages/viewpage.action?pageId=906457742)，  [feedas代码串讲](http://wiki.baidu.com/pages/viewpage.action?pageId=876128506)， [feedas&feedproxy串讲问题](http://wiki.baidu.com/pages/viewpage.action?pageId=1240957350)

3. 如何评价广告投放的效果？

需要明确，投放的广告是品牌广告还是效果广告，品牌广告重心是品提升品牌知名度、美誉度、忠诚度，其效果的衡量为长期效应，代表的行业有快消、奢侈品、汽车等；效果广告重心则在直接的投放效果，目的在于转化，比如注册、下载、购买等，通常以短期回报来衡量，代表行业有游戏、电商等；
一般来说，广告的考核指标分为三类：基础类指标，效果类指标，品牌类指标。一般来说，效果类广告评估关注基础指标+效果指标，品牌类广告评估关注基础指标+品牌类指标

- 基础指标：CPM(Cost Per Mille/Impression)每千人成本； CPC(Cost Per Click/Click-through)点击成本；CTR(Click Through Rate) 点击率
- 效果指标：CPA(Cost Per Action)每行动成本，比如app每下载一次花费的成本；ROI(Return On Investment)投资回报率，即某一周期内，广告投放收回的价值占投入的百分比
- 品牌类指标：展示(Impression)广告获得了多少次的展示；到达(Reach)广告触达了多少人； 独立访客数(Unique Visitor 数量) 指在特定时间内访问页面设备总数; 频次(Frequency)一条素材要控制在对每个UV的曝光频次不超过3次； 互动率(Engagement Rate)用于衡量广告投放中用户在广告素材或者站内（网站或APP）的交互度和参与度。

4. 在线召回和离线召回的区别？为什么要设置在线召回和离线召回？

离线召回使用的是用户历史行为数据，在线召回使用的是用户实时的行为数据。在线召回根据用户的线上行为实时反馈，快速跟踪用户的偏好，能够缓解/解决用户冷启动问题，实时性更好。离线召回使用历史行为数据，能够从长期的行为特征把握用户的偏好，增加召回的准确性和有效性。

5. 什么情况下需要调整cpm和ctr门槛，设置门槛的目的是什么？

设置ctr门槛是为了提升广告的质量和效益，而设置cpm门槛的目的，是在保证了点击率的前提下，最大化广告平台的收入。另外从系统的角度，门槛阈值本身设置的目的之一在于大量数据样本情况中，在不影响业务效果前提下筛除掉非目标样本，使得后续模块的计算成本降低。

6. 客户层面是怎么表达的，深度转化和浅层转化的区别是什么，设置二者的目的是什么？

浅层转化率和深层转化率达区别在于其场景，浅层转化是用户从流量平台到达广告主方，深层转化则为在广告主方进行了付费或者较为深度的交互的行为，二者都是ocpx广告的优化目标，设置二者的目的是为了更好的刻画广告主的收益，提升广告投放的效果。

参考：[深度转化现状整理](http://wiki.baidu.com/pages/viewpage.action?pageId=1275389788)， 

7. cpm1和ecpm的值大小有什么关系？

cpm1与ecpm分别表示千次检索收益与千次曝光收益，计算公式分别为：cpm1 = charge / pv * 1000，ecpm = charge / eshow * 1000 。不难看出，两个指标的计算区别在于分母不同，所以，比较pv与eshow的大小即可得知cpm1与ecpm的大小。在多次请求中cpm1与ecpm没有恒定的大小关系，在N次检索中，pv = N，eshow = N * pvr * asn * 曝光率，故有以下三种情况：

- 当 pvr * asn * 曝光率 < 1，即平均每次检索对应广告曝光条数小于1时，此时有cpm1 < ecpm；
- 当 pvr * asn * 曝光率 = 1 时，有cpm1 = ecpm
- 当 pvr * asn * 曝光率 > 1 时，有cpm1 > ecpm

asn表示平均展现条数，计算公式为展现条数 / 出广告的检索量

参考：[广告术语](http://wiki.baidu.com/pages/viewpage.action?pageId=1316532027) 
展现对应着show，曝光对应着eshow，展现是指被feedas传回给端上的广告，但不一定会被用户看到，曝光是指用户实际看到的广告，广告有展现并不一定有曝光。

8. 在smart_bid中，有很多个调整系数，哪个系数优先级最高？

smart_bid目的：在smart_bid阶段，主要完成bid_ratio/ocpx_bid_ratio的调整，计算出最终调整后的bid及cpm。

user_smart_bid: 跳过cpm和ocpc二阶段广告，根据src_id&mt、是否是程序化创意广告、当前hour调整bid_ratio，进而调整bid、更新计费排序score
plugin_name : user_smart_bid
alive : 1
[...plugin_params]
enable_virtual_mt_smart_bid : 0
is_set_src_bid_ratio : 0
bid_ratio : 1.0
以下广告类型不走：cpm、ocpc的暗投和明投、bes明示单选、ocpc二阶。
初始的adv->bid_ratio=1
rta溢价系数：1-2，上下界。
针对cpc广告：

1. 按照cuid的tag和user source个性化调整bid 系数
2. 按小时级别调整广告的bid_ratio -- 但是词表内容为空，实际不生效

ocpc_bid：只对ocpc二阶段广告出价进行调整，读取配置并设置ocpc_bid_ratio，根据分样式、分os、激励视频、程序化创意等调整ocpc_bid_ratio，进而调整bid、更新计费排序score
plugin_name : ocpc_bid
alive : 1
[...plugin_params]
ctrq_thrd:0
calibration_ratio:1.0
ocpc_bid_ratio:1.35
ann_ocpc_bid_ratio:1.1
enable_ocpc_virtual_mt_smart_bid : 1
minging_type_default_bid_ratio : 1.1
minging_type_relq_bid_ratio : 1.1
minging_type_attention_bid_ratio : 1.05
is_feed_async_ocpc_bid_ratio:0
feed_async_ocpc_bid_ratio:1
feed_async_ocpc_bid_ratio_ios:1
feed_async_ocpc_bid_ratio_android:1
promote_quantity_bid_ratio : 1.2
ocpc_lab_mode_bid_ratio : 1.15
ocpc_max_minbid : 35
ocpm_bid_ratio : 1.0
user_ocpc_bid_adjust_ratio_w : 1.0
is_need_user_qratio_adjust : 1

1. 按照cvr调价策略 -- 没开
2. roiq估不准，所以限制max
3. 按照ocpc反馈系数（默认2）进行调价, 按照转换类型 -- 1.35
4. 按照virtual_mt_bid -- 1.1
5. 激励视频 -- 1.2
6. rta溢价系数：1-2，上下界。
7. bes的调整
8. 游戏视频 -- 0.8

参考：[smart_bid策略详解](http://wiki.baidu.com/pages/viewpage.action?pageId=1416120980)， [Smart_bid阶段](http://wiki.baidu.com/pages/viewpage.action?pageId=950796354)

--------------------

9. 行业屏蔽是在as、proxy、bs哪个模块做的，为什么？

答：由于广告屏蔽里面的行业屏蔽涉及的广告来源途径众多，比如竞价广告、闪投广告、GD广告等，目前feed proxy并没有覆盖所有来源等广告，存在BesRta直连as，所以行业屏蔽应该放在as模块实现，以达到全流量行业屏蔽的效果。

10. 说下pv内的去重和pv间的去重，二者的区别是什么？

名词解释：pv(page view) 指页面浏览量， 用户统计站点或app某页面或板块访问次数。
pv内的去重: 一次页面浏览可能有多个广告位展示，同一个广告主可能会买同一类型的多个广告，因此一个广告位可能有同一个广告主多个的广告，因此需要在feedproxy和feedas进行去重，feedproxy是userid等维度，过滤出6个广告左右；feedas是userid、planid、ideaid等多个维度，最终过滤出一个广告。具体来说，as策略的de_dup插件是用于广告的去重，主要包括user_id、planid、unitid，subject、branch、mt、title、trade2。按user_id、planid、unitid维度去重是防止同一个广告主自己的广告之间相互顶价（导致广告主多付钱；按subject、branch、mt、title、trade2是防止给用户的不同广告位推相同的广告，提高用户体验。

pv间的去重：实际上是频控，针对一段时间内的pv展示的广告进行约束。频控相关内容参见问题2的解答。

pv内的去重和pv间的去重的区别在于施加约束的对象，pv内的去重针对的是一次pv中涉及的广告，而pv间的去重针对的是一段时间内的所有涉及的广告。

11. 自然属性，商业属性，对应不同的时长

自然属性具有先天性，一旦形成基本不变，比如年龄、性别、血型；社会属性、商业属性则是后天形成的，相对稳定，比如职业、婚姻状态，消费偏好等等

12. 总结和对比下各个用户信息获取模块

- UasProcessModule：uas返回数据有性别，年龄，query_profile,兴趣，意图，其他属性，比如收入水平，所在行业，星座，应用列表，教育水平，消费水平等，替换掉upin返回的相应属性。

- UserCenterProcessModule：解析频控session，替换掉upin的freq_feed_session；解析show session；解析expose_duration；解析expose_slot_duration

- UpinProcessModule：获取用户画像，包括性别，年龄，query_profile,兴趣，意图，其他属性，比如收入水平，所在行业，星座，应用列表，教育水平，消费水平，用户历史搜索过的query，解析asplog得到最近8h用户浏览过的广告的相关信息，ip定位，cuid常驻地点，cookie常驻地点等

- KaiwuProcessModule：获取意图词ei_query_list，user_profiles（从中解析得到kaiwu_interest_list），user_locations，lt_intent_crowds（用户意图的集合），app_info_list，looklike_crowds（用户类似人群的集合），kaiwu_education，pa_rt_tags 闪投rt tag列表，pa_it_tags 闪投it tag列表，pa_kt_list 闪投定制的ei->query列表，其他属性，比如收入水平，所在行业，星座，应用列表，教育水平，消费水平等。

13. 在系统中有多个bs，这些bs有什么区别？放在一个模块里可以么？

目前系统中有四个bs，分别是feedbs：基础检索模块；interestbs：兴趣bs；feedadpuls：离线挖掘广告检索模块，其中离线挖掘广告检索包括尤里卡(xbox)、ANN检索(annserver)、EE(离线本地词表)等。抽离出单独的离线挖掘广告检索服务后，解耦了feedbs模块中繁重的离线和在线检索逻辑； besbs：外部流量（联盟）。

bs模块负责根据用户特征、检索的特征获取一批广告，需要根据不同触发通道对不同来源广告库进行离线和在线的广告召回，放在一个模块里，该模块的逻辑结构实现难度和复杂度较大，响应速度和延时无法保证。

14. 广告日志

asp日志：为请求级别日志，记录请求、各模块耗时、广告级别结果等信息（量级大）。
notice日志：记录请求、广告过滤、广告基本信息、实验相关信息及部分q值信息。
fensui：负责记录交互耗时和函数运行耗时等信息。
计费日志：广告点击会跳转到计费服务器，从而记录计费日志。
afd日志：AFD相关日志。
als日志：记录广告曝光和cpm广告点击等信息，会用于模型训练或数据分析

15. 消费控制是怎么样的？

消费控制：在投放过程中，系统会统计plan下 budget_id里的消费/plan总消费 是否超出控制比率，超出则广告不再曝光，针对单独的渠道进行费用的控制。

举例：若规定手百消费占40%，bes消费占60%
plan总消费：为目前已经产生的消费（手百+bes）
budget_id里的消费(如：bes部分)，已经超过了plan总消费的60%，则广告被过滤。直到手百部分继续消费（一般手百消费不设限），增大plan总消费，整体比率低于阈值，bes部分才可继续消费(budget_id下包括了很多cmatch)
消费控制: constant_speed_delivery函数中做了消费控制：
  / / 计划投放期间的前n分钟时间内限制投放概率
  show_prob = PD()->global_params_conf->show_prob_first_n_minutes();
  / / 超出小时级别的消费阈值，停止展现，diff_value>epsilon，停止展现
  diff_value = plan_consume(广告计划整体消费) * 1.0 / control_budget(广告计划预算)-
			           pass_exp_consume_ratio(一天中已经过去的可投放时间区间预计消费比例和当前投放时间段的预计消费比例) / all_day_exp_consume_ratio(全天所有可投放时间段的预计消费总和) * coefficient_a;

16. 业务端逻辑是怎么样的？

一个广告在业务端主要有四个层级，分别是用户，广告计划，单元，创意+选词+意图层级。
用户层级是广告主设置的账户，能够与其公司相对应，可以用来设置余额、预算等信息
计划层级，管钱，控制分配计划的预算，
单元层级，控制找什么样的人，怎么找人，最合适的产品给合适的人
推广创意层级，让他看什么，引入了创意文案、落地页信息、图片、创意组件等

每个计划对应着四个目标中的一个：
网站链接：适用于网站推广的广告主，提升页面的到访，进而在页面产生咨询、留下联系方式等销售线索
应用推广：适用于应用推广诉求的广告主，提升应用的下载、安装、激活、付费等
商品目录：适用于拥有众多商品且已对接百度商品中心的广告主，如零售平台、汽车平台、旅游平台、房产平台等
电商店铺：适用于拥有成熟电商店铺的广告主，希望直达店铺、提升销量，目前已支持淘宝、天猫、京东、拼多多、1688等平台的店铺
门店推广：适用于拥有线下门店的广告主，希望提升门店卡券分发、线索收集、线下到访等营销效果，
设定目标的作用包括制作广告过程中，系统将帮忙找到达成该目标最佳的产品功能并突出显示，节省广告制作时间并制作出更加出色的广告。在计划设置中包含计划的预算(自定义预算单位为元/天)，推广的日期，投放的时段，投放的方式(实际上是速度，包含三种，标准投放，均衡投放-匀速投放，展现优先-加速投放)

17. simulation

随着越来越多的流量类型接入，检索端的业务逻辑日益精细。如何高效、准确的获取合理化的策略漏斗机制显得尤为重要，simulation系统将基于离线存储的模拟日志，复现真实请求和粗选广告队列，线下模拟策略漏斗机制，从根本上解决策略迭代效率受限和人工调参周期长的问题，进而提升系统的整体变现能力。Simulation系统旨在实现**策略效果预估**和**参数智能寻优**两大核心功能。主要包含两大模块：

- Online Extractor模块
  ​      (1)  解耦广告数据准备阶段和广告策略漏斗，为粗选广告队列中的广告进行日志埋点
  ​      (2)  在策略漏斗处理完成后，采集模拟日志上传至minos平台，完成基础数据流的建设
- Offline Simulator模块
  ​      (1)  Simulator解析离线模拟日志，构造粗选广告队列，作为线下策略模拟的输入数据
  ​      (2)  线下模拟基于feedas漏斗策略，挂载待预估策略插件和待调参配置
  ​      (3)  策略模拟输出结果：粗选广告数据、策略模拟结果数据，以供后续效果评估系统使用

参考：[Feedas Simulation系统](http://wiki.baidu.com/pages/viewpage.action?pageId=325901264)