## Phase概览

*   **涉及模块**
    *   非交互类：PostProcessModule

*   **代码路径**：`feedas/framework/`

## 执行顺序

*   **非交互类**

    基本顺序为：prepare_context => `handle_data` => debugpf

    |             | handle_data                                                  | 作用                      |
    | :---------- | :----------------------------------------------------------- | :------------------------ |
    | PostProcess | ① 多队列广告截断，累计不超过tuncate_num<br>② 打包组件样式<br>③ 包装计费串：对每个src_id分别计算i、j、k域，优先计算k域<br>④ 添加组合样式计费串，在这之上添加CPC和CPV计费串 | 截断&打包广告、填充计费串 |
    
    >   *   计费串由i,j,k域组成
    >       *   i：广告层级信息，如planid等
    >       *   j：包含计费名称，为跳转链接
    >       *   k：统计各种扩展信息
    >

## handle_data

*   **运行样式策略插件**：`run_style_process_pulgins`
*   **处理样式相关**：`pack_newstyle`
*   **包装点击串**：主计费串`pack_rcv_url`
*   **组件样式添加cpm/cpv计费串**：`pack_newstyle_rcv_url`
*   **新建分区域计费的计费串**: `handle_area_rcv_url`
*   **GD和cpc广告逻辑处理**: `geed_force_occupy_cpc_advertise`
*   **生产全流量id**: `generate_accurate_ovlexp_id`
