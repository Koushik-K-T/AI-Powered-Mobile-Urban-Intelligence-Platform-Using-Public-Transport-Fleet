// ==========================================
// UrbanPulse - Smart Road Intelligence
// Prototype Simulation
// ==========================================

let simulationRunning = false;
let simulationTimer = null;
let eventCount = 0;
let map = null;
let busMarker = null;
let potholeMarker = null;


// ==========================================
// MAP INITIALIZATION
// ==========================================

function initializeMap() {

    const mapElement = document.getElementById("map");

    if (!mapElement) {
        console.error("Map element not found.");
        return;
    }

    // Prevent creating the map more than once
    if (map !== null) {
        return;
    }

    map = L.map("map").setView([12.9716, 77.5946], 13);

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution: "&copy; OpenStreetMap contributors"
        }
    ).addTo(map);


    // Bus marker
    const busIcon = L.divIcon({
        className: "bus-map-marker",
        html: "🚌",
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });

    busMarker = L.marker(
        [12.9716, 77.5946],
        { icon: busIcon }
    ).addTo(map);

    busMarker.bindPopup(
        "<b>UrbanPulse Bus 12</b><br>GPS Connected"
    );


    // Initial pothole marker
    const potholeIcon = L.divIcon({
        className: "pothole-map-marker",
        html: "⚠️",
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });

    potholeMarker = L.marker(
        [12.9750, 77.6000],
        { icon: potholeIcon }
    );

    // Don't show the pothole until simulation starts
}


// ==========================================
// START SIMULATION
// ==========================================

function startSimulation() {

    if (simulationRunning) {
        return;
    }

    simulationRunning = true;

    const video = document.getElementById("roadVideo");

    const detectionStatus =
        document.getElementById("detectionStatus");

    const issueCount =
        document.getElementById("issueCount");

    const highCount =
        document.getElementById("highCount");

    const detectionBox =
        document.getElementById("detectionBox");

    const priorityBadge =
        document.getElementById("priorityBadge");

    const priorityMessage =
        document.getElementById("priorityMessage");

    const locationText =
        document.getElementById("locationText");

    const sightings =
        document.getElementById("sightings");

    const progressBar =
        document.getElementById("progressBar");

    const startBtn =
        document.getElementById("startBtn");


    // Change button
    if (startBtn) {
        startBtn.textContent = "● Simulation Running";
        startBtn.disabled = true;
    }


    // Detection status
    if (detectionStatus) {
        detectionStatus.textContent =
            "AI Detection Active";
    }


    // Start video
    if (video) {
        video.currentTime = 0;

        video.play().catch(function () {
            console.log("Video requires manual play.");
        });
    }


    // First detection after 1.5 seconds
    simulationTimer = setTimeout(function () {

        eventCount = 1;

        if (issueCount) {
            issueCount.textContent = "1";
        }

        if (highCount) {
            highCount.textContent = "0";
        }

        if (sightings) {
            sightings.textContent = "1";
        }

        if (detectionStatus) {
            detectionStatus.textContent =
                "Pothole detected";
        }

        if (detectionBox) {
            detectionBox.classList.remove("hidden");
        }

        if (priorityBadge) {
            priorityBadge.textContent = "MEDIUM";
            priorityBadge.className =
                "priority-badge medium";
        }

        if (locationText) {
            locationText.textContent =
                "Bengaluru • 12.9750, 77.6000";
        }

        if (priorityMessage) {
            priorityMessage.textContent =
                "Pothole detected by Bus 12. Road inspection recommended.";
        }

        if (progressBar) {
            progressBar.style.width = "60%";
        }


        // Add pothole to map
        if (potholeMarker && map) {
            potholeMarker.addTo(map);

            potholeMarker.bindPopup(
                "<b>Pothole Detected</b><br>" +
                "Bus 12<br>" +
                "Confidence: 91%"
            ).openPopup();
        }


        // Add event
        addDetectionEvent(
            "Pothole detected",
            "Bus 12 • Confidence 91%"
        );


    }, 1500);


    // High priority after 5 seconds
    setTimeout(function () {

        if (!simulationRunning) {
            return;
        }

        if (highCount) {
            highCount.textContent = "1";
        }

        if (sightings) {
            sightings.textContent = "2";
        }

        if (priorityBadge) {
            priorityBadge.textContent = "HIGH";
            priorityBadge.className =
                "priority-badge high";
        }

        if (priorityMessage) {
            priorityMessage.textContent =
                "High-priority road issue detected. Municipal response recommended.";
        }

        if (progressBar) {
            progressBar.style.width = "90%";
        }

        addDetectionEvent(
            "High priority road issue",
            "Bus 12 • Municipal action recommended"
        );

    }, 5000);
}


// ==========================================
// ADD DETECTION EVENT
// ==========================================

function addDetectionEvent(title, details) {

    eventCount++;

    const eventList =
        document.getElementById("eventList");

    const eventCounter =
        document.getElementById("eventCounter");


    if (eventCounter) {
        eventCounter.textContent =
            eventCount + (eventCount === 1 ? " event" : " events");
    }


    if (!eventList) {
        return;
    }


    // Remove empty message
    const emptyEvent =
        eventList.querySelector(".empty-event");

    if (emptyEvent) {
        emptyEvent.remove();
    }


    const event = document.createElement("div");

    event.className = "event-item";

    event.innerHTML = `
        <div class="event-icon">⚠</div>

        <div class="event-content">
            <strong>${title}</strong>
            <span>${details}</span>
        </div>

        <div class="event-time">
            NOW
        </div>
    `;

    eventList.prepend(event);
}


// ==========================================
// RESET SIMULATION
// ==========================================

function resetSimulation() {

    simulationRunning = false;

    if (simulationTimer) {
        clearTimeout(simulationTimer);
        simulationTimer = null;
    }


    const video =
        document.getElementById("roadVideo");

    const detectionStatus =
        document.getElementById("detectionStatus");

    const issueCount =
        document.getElementById("issueCount");

    const highCount =
        document.getElementById("highCount");

    const detectionBox =
        document.getElementById("detectionBox");

    const priorityBadge =
        document.getElementById("priorityBadge");

    const priorityMessage =
        document.getElementById("priorityMessage");

    const locationText =
        document.getElementById("locationText");

    const sightings =
        document.getElementById("sightings");

    const progressBar =
        document.getElementById("progressBar");

    const eventList =
        document.getElementById("eventList");

    const eventCounter =
        document.getElementById("eventCounter");

    const startBtn =
        document.getElementById("startBtn");


    // Reset numbers
    if (issueCount) {
        issueCount.textContent = "0";
    }

    if (highCount) {
        highCount.textContent = "0";
    }

    if (sightings) {
        sightings.textContent = "0";
    }


    // Reset detection
    if (detectionStatus) {
        detectionStatus.textContent =
            "Waiting for simulation...";
    }

    if (detectionBox) {
        detectionBox.classList.add("hidden");
    }


    // Reset priority
    if (priorityBadge) {
        priorityBadge.textContent = "WAITING";
        priorityBadge.className =
            "priority-badge waiting";
    }

    if (locationText) {
        locationText.textContent =
            "Waiting for detection";
    }

    if (priorityMessage) {
        priorityMessage.textContent =
            "No road issue detected yet.";
    }

    if (progressBar) {
        progressBar.style.width = "0%";
    }


    // Reset events
    eventCount = 0;

    if (eventCounter) {
        eventCounter.textContent = "0 events";
    }

    if (eventList) {
        eventList.innerHTML = `
            <div class="empty-event">
                <div>◌</div>
                <p>Start the simulation to see detection events.</p>
            </div>
        `;
    }


    // Reset button
    if (startBtn) {
        startBtn.textContent = "▶ Start Simulation";
        startBtn.disabled = false;
    }


    // Reset video
    if (video) {
        video.pause();
        video.currentTime = 0;
    }


    // Remove pothole marker
    if (potholeMarker && map) {
        map.removeLayer(potholeMarker);
    }

    // Reset map view
    if (map) {
        map.setView([12.9716, 77.5946], 13);
    }
}


// ==========================================
// PAGE LOAD
// ==========================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        initializeMap();

        console.log(
            "UrbanPulse initialized successfully."
        );
    }
);