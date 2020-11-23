(function(){
    var plugin = function(hook) {
        hook.doneEach(function() {
            window._hmt = window._hmt || []
            var hm = document.createElement("script")
            hm.src = "https://hm.baidu.com/hm.js?" + window.$docsify.bdStatId
            document.querySelector("article").appendChild(hm)
        })
    }
    var plugins = window.$docsify.plugins || []
    plugins.push(plugin)
    window.$docsify.plugins = plugins
})()