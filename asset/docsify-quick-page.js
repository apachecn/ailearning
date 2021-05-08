document.addEventListener('DOMContentLoaded', function() {	
	var prevBtn = document.createElement("div")
	prevBtn.id = "prev-page-button"
	document.body.appendChild(prevBtn)
	var nextBtn = document.createElement("div");
	nextBtn.id = "next-page-button"
    document.body.appendChild(nextBtn)

    var links = null
	var linkMap = null
	var getCurIdx = function() {
		if (!links) {
			links = Array
				.from(document.querySelectorAll(".sidebar-nav a"))
				.map(x => x.href)
			linkMap = {}
			links.forEach((x, i) => linkMap[x] = i)
		}
		
		var elem = document.querySelector(".active a")
		var curIdx = elem? linkMap[elem.href]: -1
		return curIdx
	}

	prevBtn.addEventListener('click', function () {
		if (!document.body.classList.contains('ready'))
			return
		var curIdx = getCurIdx()
		location.href = curIdx == -1? 
			links[0]: 
			links[(curIdx - 1 + links.length) % links.length]
		document.body.scrollIntoView()
	}, false)
	
	nextBtn.addEventListener('click', function () {
		if (!document.body.classList.contains('ready'))
			return
		var curIdx = getCurIdx()
		location.href = links[(curIdx + 1) % links.length]
		document.body.scrollIntoView()
	}, false)
})