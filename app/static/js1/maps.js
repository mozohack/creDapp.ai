$(function() {
            var plants = [{
                name: 'Australia',
                coords: [-15.274398, 133.775136],
                status: 'closed',
            }, {
                name: 'China',
                coords: [45.274398, 100.775136],
                status: 'closed',
            }, {
                name: 'India',
                coords: [30.274398, 78.775136],
                status: 'closed',
            }, {
                name: 'Brazil',
                coords: [-1.274398, -50.775136],
                status: 'closed',
            }, {
                name: 'Canada',
                coords: [65.090240, -110.712891],
                status: 'closed',
            }, {
                name: 'USA',
                coords: [45.090240, -100.712891],
                status: 'closed',
            }, {
                name: 'Algeria',
                coords: [35.9030599, 2.4213693],
                status: 'closed'
            }, {
                name: 'Greenland',
                coords: [78.9030599, -38.4213693],
                status: 'closed'
            }, {
                name: 'South Africa',
                coords: [-20.9030599, 23.4213693],
                status: 'closed',
                text: 'heelo'
            }, {
                name: "Russia",
                coords: [65.9030599, 95.4213693],
                status: 'closed'
            }];
            new jvm.Map({
                container: $('#map'),
                map: 'world_merc',
                markers: plants.map(function(h) {
                    return {
                        name: h.name,
                        latLng: h.coords
                    }
                }),
                labels: {
                    markers: {
                        render: function(index) {
                            return plants[index].name;
                        },
                        offsets: function(index) {
                            var offset = plants[index]['offsets'] || [0, 0];
                            return [offset[0] - 7, offset[1] + 3];
                        }
                    }
                },
                zoomOnScroll: false,
                series: {
                    markers: [{
                        attribute: 'image',
                        scale: {
                            'closed': 'images1/map-marker.png',
                            'activeUntil2018': 'images1/map-marker.png',
                            'activeUntil2022': 'images1/map-marker.png'
                        },
                        values: plants.reduce(function(p, c, i) {
                            p[i] = c.status;
                            return p
                        }, {}),
                    }]
                }
            });
        });