## Phase概览

*   **涉及模块**
    *   非交互类：ResponseProcessModule

*   **代码路径**：`feedas/framework/`

## 执行顺序

*   **非交互类**

    基本顺序为：prepare_context => `handle_data` => debugpf

|                 | handle_data                                                  | 作用              |
| :-------------- | :----------------------------------------------------------- | :---------------- |
| ResponseProcess | ① 设置请求级别asplog，将各模块耗时写入<br/>② 打包结果，返回给asp的idl | 打包返回数据给fmp |

## handle_data

1.  **设置请求级别asplog**（query级别）
    *   `set_adv_not_true_view_filter`: 标识没有真正曝光的winfo、ideaid、planid、userid、title_sign、brand_sign、subject
    *   `set_query_asp_log`
    *   `set_asplog_mixer_total`: 记录所有模块耗时，打印到asplog
2.  **打包请求级别asp_res**
    *   `pack_asp_res_query_level`
3.  **打包广告级别结果和asplog**
    *   `pack_asp_res_adv_level`
4.  **打pb级别结果idl**
    *   `pack_realtime_asp_res_pb`
    *   pb级别是什么意思？
5.  **打包结果idl**
    *   `pack_asp_res_idl`
6.  **指标测试日志，线上不会打开**：`write_index_log`

### set_query_asp_log

*   设置相关字段：fcstat、router_tag、ui_template等
*   遍历厂商caid_company_group，设置厂商字段
*   设置bes的media和scene
*   个推落日志：第一次点击的时间 + 设置用户活跃天数
*   相似人群落日志
