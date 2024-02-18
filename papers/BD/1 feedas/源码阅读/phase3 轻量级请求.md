## Phase概览

*   **涉及模块**

    *   非交互类 LiteGoldengateSendProcessModule

        >   作用：轻量化检索框架，使后面金门response处理和用户特征阶段可以并行处理

    *   非交互类 LiteIntentServiceSendProcessModule

        >   请求意图中台

    *   交互类 PaBdrpProcessModule

        >   Bdrp: 百度分布式redis平台，提供高性能KV存储
        >
        >   如果是百度的流量，则可以发送加密的流量设备号，在bdrp中进行检索设备号绑定的创意列表
        >
        >   实质是来自京东的广告，京东广告部负责根据加密的设备号获取行为数据进行广告检索，返回idealist，写入bdrp（跳过winfo层级，直接触发idea）
        >
        >   bdrp可以缓存4h，若4h无更新，则由jd-proxy进行过期设备号收集，请求jd，将最新数据写入bdrp
        >
        >   ref：[pafeedbs模块代码详解](http://wiki.baidu.com/pages/viewpage.action?pageId=809454390)

*   **代码路径**：`feedas/framework/`

*   **执行顺序**

    >   非交互类按照index先后执行，则Req先，Lite后

    *   **Module初始化 => 进程级别数据初始化**
        *    `LiteGold::DECLEAR_MODULE_PROC_DATA` : 创建对象
        *    `LiteIntent::DECLEAR_MODULE_PROC_DATA`：创建对象
        *    `Pa::DECLEAR_MODULE_PROC_DATA`：创建对象
    *   **QueryContext初始化 => 线程级别数据初始化**
        *   `LiteGold::DECLEAR_MODULE_QUERY_CONTEXT`: 清空原有数据
        *   `LiteIntent::DECLEAR_MODULE_QUERY_CONTEXT`: 清空原有数据
        *   `Pa::DECLEAR_MODULE_QUERY_CONTEXT`: 如果需要`move_pa_bdrp_to_upin_phase`，就进行bdrp初始化，清空原有数据
    *   **交互类前期**
        *   `Pa::create_rpc_channel`：GO_ON
        *   [`Pa::prepare_request`](#Pa::prepare_request)
        *   `Pa::send_request`: GO_ON
    *   **非交互类**
        *   LiteGold: prepare_context无操作 => [`handle_data`](#LiteGold::handle_data) => debugpf
        *   LiteIntent: prepare_context => [`handle_data`](#LiteIntent::handle_data) => debugpf
    *   **交互类后期**
        *   `Pa::wait_response`
        *   [`Pa::handle_response`](#Pa::handle_response)
        *   `Pa::close_rpc_channel`

## LiteGoldengateSendProcessModule

### LiteGold::handle_data

*   获取`liteGoldengateClient`

*   创建打点对象Mri_tracer

    >   打点tracer（鹰眼平台）
    >
    >   pv级指标：漏斗平台上自定义策略监控和各种流量特征覆盖率的信息
    >
    >   过滤原因监控：每个过滤原因一个checkpoint，维度不同则tracer不同
    >
    >   q值监控：q的请求、返回数量、q值

*   client异步发送请求，参数为`env`和`q_ctx`（保存了aspreq_data）

## LiteIntentServiceSendProcessModule

### LiteIntent::handle_data

*   创建鹰眼对象mri_tracer

    >   用于数据打点

*   创建线程级数据td

*   异步发送请求

## PaBdrpProcessModule

### Pa::prepare_request

*   获取当前检索环境

*   **设置请求环境`build_request_context()`**
    
    *   解析src_id
    *   解析广告槽位adslot_item => 没有槽位为非法流量
    
*   **判断流量合法**: `self_ctx->valid_pa_flow`

*   **解析device_id**

*   **解析当前广告槽位的京东流量类型**

    >   Xbuiltin

*   **异步请求bdrp**

    >   百度分布式redis平台，提供高性能KV存储

### Pa::handle_response

*   **创建鹰眼对象mri_tracer**
*   **获取当前检索环境**
*   **判断是否非法流量**
*   **创建bdrp_guard对象**
*   **等待bdrp返回**
*   **bdrp_guard.dismiss()**

