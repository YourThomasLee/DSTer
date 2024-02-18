## Phase概览

*   **涉及模块**
    *   非交互类`DataManagerModule`
*   **代码路径**：`feedas/framework/`
*   **执行顺序**
    *   Module初始化
        *   [`DECLEAR_MODULE_PROC_DATA`](#DECLEAR_MODULE_PROC_DATA) => 进程级别数据初始化
        *   [`mod->register_conf`](#register_conf)
        *   [`mod->register_dict`](#register_dict)
    *   QueryContext初始化：[`DECLEAR_MODULE_QUERY_CONTEXT`](#DECLEAR_MODULE_QUERY_CONTEXT) => 线程级别数据初始化
    *   [`q_ctx->prepare_context()`](prepare_context) => 若之前线程数据初始化失败，则此处重试初始化
    *   `mod->handle_data()` => 此处无操作

## DataManagerModule

### DECLEAR_MODULE_PROC_DATA

>   在`remix_init()`中完成，MODULE初始化

*   创建`DataManagerProcData`对象

*   `_module_proc_data->initialize()`：此处指`DataManagerProcData`的initialize

    *   初始化进程数据指针`ProcessData* pd`
    *   pd->initialize
    *   观星schema初始化: `SchemaExtractorManager`
    *   观星api初始化：`PredictorApi`

*   `module_proc_data_macro get_proc_data()`

    >   Remix框架的base_module.cpp中定义，返回当前module的`_module_proc_data`

### DECLEAR_MODULE_QUERY_CONTEXT

>   在`remix_init()`的`Seacher.init`完成
>
>   创建ModBaseQueryContext存入QueryContext

*   remix框架中定义的基方法：初始化`_status`和`_ignored`
*   创建`DataManagerQueryContex`对象
    *   创建UpstreamData对象
    *   分配内存
    *   申请内存池
*   `query_ctx->init_context()`
    *   初始化线程数据
    *   `_fengsui_inited`置为false

### handle_data

*   接收参数：enc和q_ctx
*   返回：GO_ON

### register_conf

*   在`remix_init`的时候调用
*   作用
    *   获取进程数据ProcessData
    *   检查观星schema校验是否重复

### register_dict

*   在`remix_init`的时候调用，直接GO_ON

## DataManagerProcData

### initialize

*   见[`DECLEAR_MODULE_PROC_DATA`](#DECLEAR_MODULE_PROC_DATA)

## DataQueryContext

### init_context

*   见[`DECLEAR_MODULE_QUERY_CONTEXT`](#DECLEAR_MODULE_QUERY_CONTEXT) 

### prepare_context

*   接收参数：内存池mempool
*   满足条件：线程初始化失败+允许papredictor => 初始化线程、线程数据`ThreadData`

### QueryEnv类

>   一次检索可能有多次线程切换，解决变量不匹配的问题
>
>   ref：[QueryEnv](http://wiki.baidu.com/pages/viewpage.action?pageId=898059894)

*   作用：定义应用级的QueryEnv

**【构造方法】**

*   成员变量：`_sig`、`_region`、`_yacl_context`、`_g_query_context_manager`

**【reset】**

*   如果内存池不存在，则声明一个全局内存池
*   释放资源，归还持有的内存块memblock

**【construct】**

>   构造包含正确线程级变量的上下文环境

*   设置内存池
*   内存池->set_region
*   Yacl_manager设置线程context
*   设置全局query_context_manager

**【init】**

*   捕获线程级变量：`_region`、`_yacl_context`、`_sid`、`_g_query_context_manager`
*   construct()

