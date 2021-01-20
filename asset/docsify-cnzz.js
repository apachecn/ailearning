(function(){
    var plugin = function(hook) {
        hook.doneEach(function() {
            var sc = document.createElement('script')
            sc.src = 'https://s5.cnzz.com/z_stat.php?id=' + 
                window.$docsify.cnzzId + '&online=1&show=line'
            document.querySelector('article').appendChild(sc)
        })
    }
    var plugins = window.$docsify.plugins || []
    plugins.push(plugin)
    window.$docsify.plugins = plugins
})()