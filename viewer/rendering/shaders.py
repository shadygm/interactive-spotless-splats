# Fullscreen quad shader (for gsplat texture)
QUAD_VERTEX = """
#version 330
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTex;
out vec2 vTex;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTex = aTex;
}
"""

QUAD_FRAGMENT = """
#version 330
in vec2 vTex;
uniform sampler2D uTex;
out vec4 FragColor;
void main() {
    FragColor = texture(uTex, vTex);
}
"""

# Debug geometry shader (for points and frustums)
DEBUG_VERTEX = """
#version 330
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 mvp;
uniform float uPointSize;
out vec3 vColor;
void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
}
"""

DEBUG_FRAGMENT = """
#version 330
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
"""

# Instanced frustum shader
FRUSTUM_VERTEX = """
#version 330
layout(location=0) in float aIdx;

layout(location=1) in vec3 iCenter;
layout(location=2) in vec3 iCorner0;
layout(location=3) in vec3 iCorner1;
layout(location=4) in vec3 iCorner2;
layout(location=5) in vec3 iCorner3;

uniform mat4 mvp;
uniform vec3 uColor;
out vec3 vColor;

void main() {
    int idx = int(aIdx);
    vec3 worldPos;
    if (idx == 0) worldPos = iCenter;
    else if (idx == 1) worldPos = iCorner0;
    else if (idx == 2) worldPos = iCorner1;
    else if (idx == 3) worldPos = iCorner2;
    else worldPos = iCorner3;
    gl_Position = mvp * vec4(worldPos, 1.0);
    vColor = uColor;
}
"""

FRUSTUM_FRAGMENT = """
#version 330
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
"""
