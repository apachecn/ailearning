document.addEventListener('DOMContentLoaded', function() {
	var scrollBtn = document.createElement('div')
	scrollBtn.id = 'scroll-btn'
	document.body.append(scrollBtn)
	
	window.addEventListener('scroll', function() {
		var offset = window.document.documentElement.scrollTop;
        scrollBtn.style.display = offset >= 500 ? "block" : "none";
	})
	scrollBtn.addEventListener('click', function(e) {
		e.stopPropagation();
		var step = window.scrollY / 15;
		var hdl = setInterval(function() {
			window.scrollTo(0, window.scrollY - step);
			if(window.scrollY <= 0) {
				clearInterval(hdl)
			}
		}, 15)
	})
})