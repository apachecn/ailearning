document.addEventListener('DOMContentLoaded', function() {
    var shareBtn = document.createElement('a')
    shareBtn.id = 'share-btn'
    shareBtn.className = 'bdsharebuttonbox'
    shareBtn.setAttribute('data-cmd', 'more')
    document.body.append(shareBtn)
    
    window._bd_share_config = {
        "common":{
            "bdSnsKey":{},
            "bdText":"",
            "bdMini":"1",
            "bdMiniList":false,
            "bdPic":"",
            "bdStyle":"2",
            "bdSize":"16"
        },
        "share":{}
    }
    var sc = document.createElement('script')
    sc.src = 'http://bdimg.share.baidu.com/static/api/js/share.js' + 
        '?v=89860593.js?cdnversion=' + ~(-new Date()/36e5)
    document.body.append(sc)
})