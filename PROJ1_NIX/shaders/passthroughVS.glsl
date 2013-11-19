attribute vec4 Position;
attribute vec2 Texcoords;
varying vec2 v_Texcoords;

void main(void)
{
	v_Texcoords = Texcoords;
	gl_Position = Position;
}