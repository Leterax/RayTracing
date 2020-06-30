#version 430

#define GROUP_X 1
#define GROUP_Y 1


#define TMIN 0.0
#define TMAX 1.0


// primitives
struct Sphere
{
    vec4 info;
};


struct Plane
{
    vec4 normal;
    vec4 position;
};

struct Box
{
    vec3 corner1;
    vec3 corner2;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct HitInfo
{
    vec3 position;
    float distance;
    vec3 normal;
    bool didHit;
    vec4 color;
};


layout (local_size_x = GROUP_X, local_size_y = GROUP_Y) in;

// uniform variables
layout(binding=0, rgba32f) uniform image2D framebuffer;
layout(binding=1, r32f) uniform image2D depthbuffer;

// Camera ubo
layout(binding=2, std140) uniform Camera
{
    vec4 eye, lower_left_corner, horizontal, vertical, origin;
    vec4 u,v,w;
    float lens_radius;
} camera;

// ssbo's:
layout(binding=3, std140) buffer SphereIntersectionBlock
{
    Sphere objects[];
} SphereIntersectionObjects;

layout(binding=4, std140) buffer PlaneIntersectionBlock
{
    Plane objects[];
} PlaneIntersectionObjects;

// intersection functions
// plane intersection
bool intersect(const Ray ray, const Plane plane, out HitInfo info) {

    float denom = dot(plane.normal.xyz, ray.direction);
    if (abs(denom) > 0.0001) {
        float t = dot((plane.position.xyz - ray.origin), plane.normal.xyz) / denom;
        if (t < TMAX && t > TMIN) {
            info.distance = t;
            info.position = ray.origin + ray.direction * t;
            info.normal = plane.normal.xyz;
            info.didHit = true;

            // color calculations:
            vec3 dt = info.position - plane.position.xyz;
            vec3 b = cross(plane.normal.xyz, vec3(0.5));

            float ln_dt = length(dt);
            float ln =ln_dt*length(b);
            float Dot = dot(dt, b)/ln;
            float Det = dot(plane.normal.xyz, cross(b, dt)/ln);
            float alpha = atan(Det, Dot);

            vec2 pos = floor(vec2(ln_dt * cos(alpha), ln_dt * sin(alpha)));


            float patternMask = mod(pos.x + mod(pos.y, 2.), 2.);
            info.color = vec4(vec3(patternMask/2.0 + 0.3), 1.0);
            return true;
        }
    }
    return false;

}


// sphere intersection
bool intersect(const Ray ray, const Sphere sphere, out HitInfo info) {
    vec3 oc = ray.origin - sphere.info.xyz;
    float a = dot(ray.direction, ray.direction);
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.info.w;
    float discriminant = b*b - a*c;

    if (discriminant < 0) {
        return false;
    }

    float t = (-b - sqrt(discriminant)) / a;
    if (t < TMAX && t > TMIN) {
        info.distance = t;
        info.position = ray.origin + ray.direction * t;
        info.normal = (info.position - sphere.info.xyz) / sphere.info.w / sphere.info.w;
        info.color = vec4(t/2.,t/3.,t/4.,1.);
        info.didHit = true;
        return true;
    }

    t = (-b + sqrt(discriminant)) / a;
    if (t < TMAX && t > TMIN) {
        info.distance = t;
        info.position = ray.origin + ray.direction * t;
        info.normal = (info.position - sphere.info.xyz) / sphere.info.w / sphere.info.w;
        info.color = vec4(t/2.,t/3.,t/4.,1.);
        info.didHit = true;
        return true;
    }

    return false;

}

vec4 sky_color(const Ray ray) {
    vec3 direction = normalize(ray.direction);
    float t = 0.5*(direction.y +1.);
    return vec4((1.-t)*vec3(1.)+t*vec3(0.5, 0.7, 1.0), 1.);
}


// trace over all objects
HitInfo trace(Ray ray) {
    HitInfo info;

    HitInfo bestHit;
    bestHit.distance = TMAX;
    bestHit.didHit = false;

    for (int i = 0; i < SphereIntersectionObjects.objects.length(); i++) {
        if (intersect(ray, SphereIntersectionObjects.objects[i], info)) {
            if (info.distance < bestHit.distance) {
                bestHit = info;
            }
        }
    }
    for (int i = 0; i < PlaneIntersectionObjects.objects.length(); i++) {
        if (intersect(ray, PlaneIntersectionObjects.objects[i], info)) {
            if (info.distance < bestHit.distance) {
                bestHit = info;
            }
        }
    }

    if (!bestHit.didHit) {
        // didn't hit anything :(
        bestHit.color = sky_color(ray);
        bestHit.distance = TMAX;
    }
    return bestHit;


}

void main(void) {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(framebuffer);
    if (pix.x >= size.x || pix.y >= size.y) {return;}
    vec2 pos = vec2(pix)/ vec2(size.x, size.y);

    vec3 dir = camera.lower_left_corner.xyz + pos.x*camera.horizontal.xyz+ pos.y*camera.vertical.xyz - camera.eye.xyz;

    Ray ray = Ray(camera.eye.xyz, dir);
    HitInfo hit = trace(ray);

    imageStore(framebuffer, pix, hit.color);
    imageStore(depthbuffer, pix, vec4(hit.distance));
}
