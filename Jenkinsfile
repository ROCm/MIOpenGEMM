
def rocmtestnode(variant, name, dockerfile, body) {
    def image = "miopengemm-${dockerfile}"
    def cmake_build = { compiler, flags ->
        def cmd = """
            rm -rf build
            mkdir build
            cd build
            CXX=${compiler} CXXFLAGS='-Werror' cmake ${flags} .. 
            CTEST_PARALLEL_LEVEL=4 dumb-init make -j32 check
        """
        echo cmd
        sh cmd
    }
    node(name) {
        stage("checkout ${variant}") {
            checkout scm
        }
        def docker_path = "docker/${dockerfile}.docker";
        stage("image ${variant}") {
            try {
                docker.build("${image}", ". -f ${docker_path}")
            } catch(Exception ex) {
                docker.build("${image}", "--no-cache . -f ${docker_path}")

            }
        }
        withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video') {
            timeout(time: 1, unit: 'HOURS') {
                body(cmake_build)
            }
        }
    }
}
@NonCPS
def rocmtest(m) {
    def builders = [:]
    for(e in m) {
        def label = e.key;
        def action = e.value;
        builders[label] = {
            action(label)
        }
    }
    parallel builders
}

@NonCPS
def rocmnode(name, image, body) {
    def node_name = 'rocmtest || rocm'
    if(name == 'fiji') {
        node_name = 'rocmtest && fiji';
    } else if(name == 'vega') {
        node_name = 'rocmtest && vega';
    } else {
        node_name = name
    }
    return { label ->
        rocmtestnode(label, node_name, image, body)
    }
}

rocmtest clang: rocmnode('vega', 'ubuntu') { cmake_build ->
    stage('Clang Debug') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('Clang Release') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, gcc: rocmnode('vega', 'ubuntu') { cmake_build ->
    stage('GCC Debug') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('GCC Release') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}
