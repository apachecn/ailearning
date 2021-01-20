(function(){
    var plugin = function(hook) {
        hook.doneEach(function() {
            new Image().src = 
                '//api.share.baidu.com/s.gif?r=' + 
                encodeURIComponent(document.referrer) + 
                "&l=" + encodeURIComponent(location.href)
        })
    }
    var plugins = window.$docsify.plugins || []
    plugins.push(plugin)
    window.$docsify.plugins = plugins
})()