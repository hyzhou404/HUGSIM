import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

window.ready_generate_car = false;
window.yaw = 0;

let innerWidth_half = window.innerWidth/4;
const scene = new THREE.Scene();
let camera = new THREE.PerspectiveCamera(75, innerWidth_half / window.innerHeight, 0.01, 10000);

// const right_scene = new THREE.Scene();
// const right_camera = new THREE.PerspectiveCamera(75, innerWidth_half / window.innerHeight, 0.01, 10000);
// const rightCanvas = document.getElementById('rightCanvas');


const origin = new THREE.Vector3(0, 0, 0);
const ambientLight = new THREE.AmbientLight(0x404040);
scene.add(ambientLight);

const leftCanvas = document.getElementById('leftCanvas');
const renderer = new THREE.WebGLRenderer({canvas: leftCanvas});

const resize = () => {
    innerWidth_half = window.innerWidth/4
    renderer.setSize(window.innerWidth/4, window.innerHeight);
    camera.aspect = window.innerWidth/4 / window.innerHeight;
    camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
renderer.setSize(window.innerWidth/4, window.innerHeight);

let controls = new OrbitControls(camera, renderer.domElement);
controls.minPolarAngle = Math.PI; 
controls.maxPolarAngle = Math.PI; 

const loader = new PLYLoader();
let pointCloud, pointCloudMaterial;
let point_cloud_center;

const gui = new GUI();
gui.domElement.style.left = '0.1%';
gui.domElement.style.position = 'absolute';

let totol_car_number = -1;
const cube_group = new THREE.Group();
scene.add(cube_group);
const cube_number_group = new THREE.Group();
scene.add(cube_number_group);
let confirm_flag = false;
let car_appearance_array = [];

const folder = gui.addFolder('Point Cloud Settings');
folder.add({ size: 0.001 }, 'size', 0.0001, 0.1).name('Point Size').onChange(function(value) {
    pointCloudMaterial.size = value;
});

//point cloud
const size = 0.01; 
const color = 0xffffff; 

// const camera_switch_btn = document.getElementById("camera_switch_btn");
// camera_switch_btn.addEventListener('click',camera_switch);

function camera_switch()
{
    const camera_perspective = new THREE.PerspectiveCamera(75, innerWidth_half / window.innerHeight, 0.01, 10000);
    const camera_ortho = new THREE.OrthographicCamera(innerWidth_half / -2, innerWidth_half / 2, window.innerHeight / 2, window.innerHeight / -2, 0.01, 10000);
    if (camera instanceof THREE.PerspectiveCamera) {
        camera = camera_ortho;
        controls.dispose();
        controls = new OrbitControls(camera, renderer.domElement);
        camera.position.set(origin.x, origin.y-200, origin.z);
        camera_switch_btn.textContent = 'Switch to Perspective';

        const scale = 0.5; 
        camera_ortho.left *= scale;
        camera_ortho.right *= scale;
        camera_ortho.top *= scale;
        camera_ortho.bottom *= scale;
        camera_ortho.updateProjectionMatrix();

    } else {
        camera = camera_perspective;
        controls.dispose(); 
        controls = new OrbitControls(camera, renderer.domElement);
        camera.position.set(origin.x, origin.y-200, origin.z);
        camera_switch_btn.textContent = 'Switch to Orthographic';
    }
    controls.update();
}

let keys = {
    w: false,
    a: false,
    s: false,
    d: false
};

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}


function updatePyramidFromEgo(newValue) {
    newValue = invert4(newValue);
    console.log("new_value",newValue);

    const ego_matrix = new THREE.Matrix4().fromArray([
        newValue[0], newValue[1], newValue[2],0,
        newValue[4], newValue[5], newValue[6],0,
        newValue[8], newValue[9], newValue[10],0,
        newValue[12], newValue[13], newValue[14],1
    ]);
    pyramidWireframe.matrix.copy(ego_matrix);
    pyramidWireframe.matrixAutoUpdate = false;
}
document.getElementById("reset_btn").addEventListener("click",()=>{
    let value =  [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ];
    updatePyramidFromEgo(value);
});


document.addEventListener('keydown', (event) => {
    if (event.key === 'a' || event.key === 'd' || event.key === 's' || event.key === 'w') {
        keys[event.key] = true;
    }
    updatePyramidFromEgo(window.ego);
});

document.addEventListener('keyup', (event) => {
    if (event.key === 'a' || event.key === 'd'|| event.key === 's' || event.key === 'w') {
        keys[event.key] = false;
    }
    updatePyramidFromEgo(window.ego);
});

const speed = 2;

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let isDragging = false;  
let startX, startY;
let first_point;
let first_point_chosen = false;
let arrowHelper = null; 


//camera
const pyramidGeometry = new THREE.ConeGeometry(2, 4, 4); 
pyramidGeometry.rotateY(Math.PI / 4); 
pyramidGeometry.rotateX(-Math.PI / 2); 
pyramidGeometry.translate(0,0,2);
const pyramidEdgesGeometry = new THREE.EdgesGeometry(pyramidGeometry);
const pyramidLineMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
const pyramidWireframe = new THREE.LineSegments(pyramidEdgesGeometry, pyramidLineMaterial);
scene.add(pyramidWireframe);


function find_intersection() {
    raycaster.setFromCamera(mouse, camera);

    const ray = raycaster.ray;
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

    const intersectionPoint = new THREE.Vector3();
    const intersectionDistance = ray.intersectPlane(plane, intersectionPoint);

    if (intersectionDistance !== null) {
        console.log('Intersection point:', intersectionPoint);
        // document.getElementById("point_info").classList.remove("hidden");
        return intersectionPoint;
    } else {
        console.log('No intersection with the plane');
        return null;
    }
}

renderer.domElement.addEventListener('mousedown', (event) => {
    isDragging = false; 
    startX = event.clientX;
    startY = event.clientY;
    updatePyramidFromEgo(window.ego);
});

renderer.domElement.addEventListener('mouseup', (event) => {
    isDragging = false; 
    startX = event.clientX;
    startY = event.clientY;
    updatePyramidFromEgo(window.ego);
});

renderer.domElement.addEventListener('mousemove', (event) => {
    if (!isDragging) {
        const distanceX = Math.abs(event.clientX - startX);
        const distanceY = Math.abs(event.clientY - startY);
        if (distanceX > 2 || distanceY > 2) {
            isDragging = true; 
        }
    }
    if(first_point_chosen)
    {
        if (arrowHelper) {
            scene.remove(arrowHelper);
            arrowHelper = null;
        }
        mouse.x = (event.clientX / innerWidth_half) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        let new_point = find_intersection();
        if(!new_point) return;
        const dir = new THREE.Vector3(new_point.x - first_point.x, new_point.y - first_point.y, new_point.z - first_point.z).normalize();
        arrowHelper = new THREE.ArrowHelper(dir, first_point, 10, 0xff0000, 3, 3);
        scene.add(arrowHelper);
        window.yaw = -Math.atan2(dir.z, dir.x); 
    } 
});

renderer.domElement.addEventListener('click', (event) => { 
    if (!isDragging)
    {
        if(!first_point_chosen)
        {   
            if (arrowHelper) {
                scene.remove(arrowHelper);
                arrowHelper = null;
            }
            mouse.x = (event.clientX / innerWidth_half) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            let xyz = find_intersection();
            if (!xyz) return;
            document.getElementById("x_value").innerHTML = xyz.x;
            document.getElementById("y_value").innerHTML = xyz.y;
            document.getElementById("z_value").innerHTML = xyz.z;
            const x_value = document.getElementById("x_value").innerHTML;
            const y_value = document.getElementById("y_value").innerHTML;
            const z_value = document.getElementById("z_value").innerHTML;
            const cube_geometry = new THREE.BoxGeometry(3, 3, 3);
            const cube_material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const cube = new THREE.Mesh(cube_geometry, cube_material);
            cube.position.set(x_value, y_value, z_value);   
            if (!confirm_flag && cube_group.children.length!==0) {
                cube_group.remove(cube_group.children[cube_group.children.length - 1]);
            }
            cube_group.add(cube);
            first_point_chosen = true;
            window.ready_generate_car = false;
            first_point = xyz;
        }
        else if (first_point_chosen) {
            first_point_chosen = false;
            confirm_flag = false;
            window.ready_generate_car = true;
        }
    }
});

document.getElementById("generate_btn").addEventListener("click", function () {
    if (window.ready_generate_car)
    {
        if (arrowHelper) {
            scene.remove(arrowHelper);
            arrowHelper = null;
        }
        cube_group.children[cube_group.children.length - 1].material.color.set(0x00ff00);
        confirm_flag = true;
        totol_car_number+=1;

        const x_value = document.getElementById("x_value").innerHTML;
        const y_value = document.getElementById("y_value").innerHTML;
        const z_value = document.getElementById("z_value").innerHTML;
        const font_loader = new FontLoader();
        font_loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function(font) {
            const textGeometry = new TextGeometry((totol_car_number).toString(), { 
            font: font,
            size: 6, 
            height: 0.2, 
            curveSegments: 12
            });  
            const textMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 }); 
            const textMesh = new THREE.Mesh(textGeometry, textMaterial);
            textMesh.rotateX(Math.PI / 2); 
            textMesh.position.set(x_value + 2.0, y_value, z_value);
            cube_number_group.add(textMesh); 
        });

        var table = document.getElementById("table");
        var length = table.rows.length;
        
        var row = table.insertRow(-1);
        var car_id = row.insertCell(0);
        var car_yaw = row.insertCell(1);
        var car_type = row.insertCell(2);
        var car_speed = row.insertCell(3);
        var car_kwargs = row.insertCell(4);
        var car_btn = row.insertCell(5);
        car_id.innerHTML = totol_car_number;
        car_yaw.innerHTML = `<input type='range' min='-3.14' max='3.14' step='0.01' value='${window.yaw.toFixed(2)}'>
    <span class='value-display'>${window.yaw.toFixed(2)}</span>`;
        car_speed.innerHTML = "<input type='range' min='0' max='4' step='0.1' value='2'><span class='value-display'>2</span>";
        car_type.innerHTML = "<select><option>Constant</option>\
                            <option>IDM</option>\
                            <option>Attack</option></select>";
        car_kwargs.className = "car_kwargs";
        car_kwargs.innerHTML = "None";
        car_btn.innerHTML = "<button class='update_btn'>update</button> <button class='delete_btn'>delete</button>";
        // car_btn.innerHTML = "<button onclick='myeditRow(this)'>编辑</button> <button onclick='mydeleteRow(this)'>删除</button>";
        const yawRange = row.querySelectorAll("input[type='range']")[0];
        const yawDisplay = row.querySelectorAll(".value-display")[0];
        const speedRange = row.querySelectorAll("input[type='range']")[1]; 
        const speedDisplay = row.querySelectorAll(".value-display")[1]

        yawRange.addEventListener("input", function() {
            yawDisplay.textContent = this.value;
        });
        speedRange.addEventListener("input", function() {
            speedDisplay.textContent = this.value;
        });
        const typeSelect = row.querySelector("select");
        typeSelect.addEventListener("change", function () {
            const carType = this.value;
            const carOptionsCell = row.querySelector(".car_kwargs");
            console.log(carOptionsCell)

            if (carType === "Attack") {
                carOptionsCell.innerHTML = `
                    <label for="pred_steps">pred_steps:</label>
                    <input type="number" id="pred_steps" min="1" max="30" value="20"><br>
                    <label for="ATTACK_FREQ">ATTACK_FREQ:</label>
                    <input type="number" id="ATTACK_FREQ" min="1" max="20" value="10"><br>
                    <label for="best_k">best_k:</label>
                    <input type="number" id="best_k" min="1" max="20" value="10">
                `;
            } else {
                carOptionsCell.innerHTML = "None";
            }
        });
        car_appearance_array.push(document.getElementById("car_appearance_selector").value.replace(".splat",""));
    }
});

document.addEventListener('click', (e) => {
    const btn = e.target;
    if (btn.classList.contains('delete_btn')) {
        var row = btn.parentNode.parentNode; 
        var table = row.parentNode; 
        var rows = Array.from(table.rows);
        var rowIndex = rows.indexOf(row)-1;
        cube_group.remove(cube_group.children[rowIndex]);
        cube_number_group.remove(cube_number_group.children[rowIndex]);
        car_appearance_array.splice(rowIndex, 1);
    }
});

document.getElementById("reset_btn").addEventListener("click",()=>{
    pyramidWireframe.position.z = 2;
    pyramidWireframe.rotation.x = 0;
    pyramidWireframe.rotation.y = 0;
});

async function getHeight(x, z) {
    const response = await fetch('get_height', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ x, z })
    });
    const result = await response.json();
    console.log("result",result);
    return result;
}

document.getElementById("saveConfigBtn").addEventListener("click", async () => {
    const table = document.getElementById("table");
    const rows = table.rows;
    const yamlContent = [];
    let load_HD_map = false;

    yamlContent.push("mode: easy_00");
    if (rows.length <= 1) { 
        yamlContent.push("plan_list: []");
    } else {
        yamlContent.push("plan_list:");
        for (let i = 1; i < rows.length; i++) { 
            const cells = rows[i].cells;
            const carId = cells[0].innerText;
            const carYaw = cells[1].querySelector("input").value; 
            const carType = cells[2].querySelector("select").value; 
            let car_speed = cells[3].querySelector("input").value;

            const cube = cube_group.children[i - 1];
            const x = parseFloat(cube.position.x);
            const y = -0.3;
            const z = parseFloat(cube.position.z);
            const car_appearance = car_appearance_array[i-1];
            // let y = await getHeight(x , z);
            // y = y.y + 1.4;
            let carYaw_save = -carYaw - Math.PI/2;
            yamlContent.push(`- - ${x.toFixed(2)}`);
            yamlContent.push(`  - ${z.toFixed(2)}`); 
            yamlContent.push(`  - ${y.toFixed(2)}`);
            yamlContent.push(`  - ${carYaw_save}`);
            yamlContent.push(`  - ${car_speed}`);
            yamlContent.push(`  - "${car_appearance}"`);
            yamlContent.push(`  - ${carType}Planner`);
            if (carType === "Attack") {
                const predSteps = cells[4].querySelector("#pred_steps").value;
                const attackFreq = cells[4].querySelector("#ATTACK_FREQ").value;
                const best_k = cells[4].querySelector("#best_k").value;
                yamlContent.push(`  - pred_steps: ${predSteps}`);
                yamlContent.push(`    ATTACK_FREQ: ${attackFreq}`);
                yamlContent.push(`    best_k: ${best_k}`);
            } 
            else{
                yamlContent.push(`  - {}`);
            } 
            if (carType === "IDM") {
                load_HD_map = true;
            }
        }
    }
    yamlContent.push(`load_HD_map: ${load_HD_map}`);
    yamlContent.push("start_euler:");
    yamlContent.push("- 0.0");
    yamlContent.push("- 0.0");
    yamlContent.push("- 0.0");
    yamlContent.push("start_ab:");
    yamlContent.push("- 0.0");
    yamlContent.push("- 0.0");
    yamlContent.push("start_velo: 1");
    yamlContent.push("start_steer: 0");
    yamlContent.push(`scene_name: ${scene_file.replace(".splat","")}`);
    yamlContent.push("iteration: 30000");

    const yamlString = yamlContent.join("\n");

    const blob = new Blob([yamlString], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "config.yaml";
    a.click();
    URL.revokeObjectURL(url);
});

// async function load_ply() {
//     return new Promise((resolve) => {
//         // 定义 20 种颜色
//         const colors = [
//             0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff, 0xff8000, 0xff0080, 0x80ff00,
//             0x0080ff, 0x8000ff, 0xff8080, 0x80ff80, 0x8080ff, 0xff0080, 0x00ff80, 0x0080ff, 0xff80ff,
//             0x80ffff, 0xff80ff
//         ];

//         // 创建颜色映射表
//         const colorMap = new Map();
//         for (let i = 0; i < 20; i++) {
//             colorMap.set(i, colors[i]);
//         }

//         loader.setCustomPropertyNameMapping({
//             semantic: ['semantic']
//         });

//         loader.load('semantic.ply', function (geometry) {
            
//             let semanticAttribute = geometry.attributes.semantic;
//             let colorArray = new Float32Array(semanticAttribute.count * 3);
//             let color = new THREE.Color();

//             for (let i = 0; i < semanticAttribute.count; i++) {
//                 const semanticValue = semanticAttribute.getX(i); // 获取 semantic 值
//                 const colorValue = colorMap.get(semanticValue); // 获取对应的颜色
//                 color.set(colorValue); // 设置颜色
//                 color.toArray(colorArray, i * 3); // 将颜色存储到数组中
//             }

//             // 将颜色数组添加到几何体
//             geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorArray, 3));

//             // 创建点云材质，启用 vertexColors
//             let pointCloudMaterial = new THREE.PointsMaterial({ vertexColors: true, size: 1 });

//             // 创建点云对象
//             pointCloud = new THREE.Points(geometry, pointCloudMaterial);
//             scene.add(pointCloud);

//             console.log(geometry);
//             point_cloud_center = geometry.boundingSphere.center;
//             resolve();
//         });
//     });
// }
async function load_ply() {
    return new Promise((resolve) => {
        loader.setCustomPropertyNameMapping({
            semantic: ['semantic'],
        });
        // TODO: read params
        loader.load(`/smt/${smt_file}`, function (geometry) {
            if (geometry.attributes.color !== undefined) {
                pointCloudMaterial = new THREE.PointsMaterial({ vertexColors: true });
            } else {
                pointCloudMaterial = new THREE.PointsMaterial({ color: color, size: size });
                folder.addColor({ color: '#ffffff' }, 'color').name('Point Color').onChange(function(value) {
                 pointCloudMaterial.color = new THREE.Color(value);
                });
            }
            pointCloudMaterial.size = 0.001;
            pointCloud = new THREE.Points(geometry, pointCloudMaterial);
            scene.add(pointCloud);
            point_cloud_center = geometry.boundingSphere.center;
            resolve();
        });
    });
}

async function init() {
    console.log('init');
    await load_ply();

    // const axesHelper = new THREE.AxesHelper( 500 );
    // axesHelper.position.set(point_cloud_center.x, point_cloud_center.y, point_cloud_center.z);
    // axesHelper.position.set(origin.x, origin.y, origin.z);
    // scene.add( axesHelper );

    camera.position.set(origin.x, origin.y-200, origin.z);
    camera.lookAt(origin.x, origin.y, origin.z);
    controls.target.set(origin.x, origin.y, origin.z);
    controls.update();
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    // const direction = new THREE.Vector3();
    // camera.getWorldDirection(direction);
    // const right = new THREE.Vector3();
    
    // if (keys.a) {
    //     const sideDirection = direction.cross(camera.up).normalize();
    //     camera.position.add(sideDirection.multiplyScalar(speed));
    //     controls.target.add(sideDirection.multiplyScalar(speed));
    // }
    // if (keys.d) {
    //     const sideDirection = direction.cross(camera.up).normalize();
    //     camera.position.sub(sideDirection.multiplyScalar(speed));
    //     controls.target.sub(sideDirection.multiplyScalar(speed));
    // }
    controls.target.set(origin.x, origin.y, origin.z);
    controls.update();
}

console.log("hello world");
init();