
/* Usage:

nvcc -m64 -shared -arch=sm_20 -o libtestkernel.so  -Xcompiler -fPIC test_kernel.cu
 
/usr/bin/gcc-4.4 -Wall -shared -fPIC -o testcustomC.so -I/usr/include -lluaT -L. -ltestkernel testcustomC.c


*/

#include <lua.h>                               /* Always include this */
#include <lauxlib.h>                           /* Always include this */
#include <lualib.h>                            /* Always include this */
#include <stdio.h>


#include "test_kernel.h"
 
static int isquare(lua_State *L){              /* Internal name of func */
	float rtrn = lua_tonumber(L, -1);      /* Get the single number arg */
	printf("Top of square(), nbr=%f\n",rtrn);
	lua_pushnumber(L,rtrn*rtrn);           /* Push the return */
	return 1;                              /* One return value */
}
static int icube(lua_State *L){                /* Internal name of func */
	float rtrn = lua_tonumber(L, -1);      /* Get the single number arg */
	printf("Top of cube(), number=%f\n",rtrn);
	
	entry();

	lua_pushnumber(L,rtrn*rtrn*rtrn);      /* Push the return */
	return 1;                              /* One return value */
}

/* Register this file's functions with the
 * luaopen_libraryname() function, where libraryname
 * is the name of the compiled .so output. In other words
 * it's the filename (but not extension) after the -o
 * in the cc command.
 *
 * So for instance, if your cc command has -o power.so then
 * this function would be called luaopen_power().
 *
 * This function should contain lua_register() commands for
 * each function you want available from Lua.
 *
*/
int luaopen_testcustomC(lua_State *L){
	lua_register(
			L,               /* Lua state variable */
			"square",        /* func name as known in Lua */
			isquare          /* func name in this file */
			);
	lua_register(L,"cube",icube);
	//lua_register(L,"entry", entry);
	//lua_register(L, "cuda_GMRESfunc", cuda_GMRESfunc);
	//entry();

	return 0;
}