## Phase概览

*   **涉及模块**
    *   非交互类：QueryProcessModule

*   **代码路径**：`tools/src/`

## 执行顺序

*   **非交互类**

    基本顺序为：prepare_context => `handle_data` => debugpf

|              | handle_data                                                  | 作用          |
| :----------- | :----------------------------------------------------------- | :------------ |
| QueryProcess | 主要函数：`process_query`<br>① 解析金门返回的数据<br>② 获取query_process的配置，为query添加索引，存入`_idx_process_query_list`<br>③ 优选query：pick_first =>取第一个; pick_random =>随机选取;pick_weight_random：加权随机 | 进行query优选 |



