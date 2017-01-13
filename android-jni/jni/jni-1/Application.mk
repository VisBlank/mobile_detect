
#
APP_ABI := armeabi armeabi-v7a
APP_ABI += armeabi-v7a-hard

#gnustl_static stlport_static gnustl_shared
APP_STL := gnustl_static
APP_CPPFLAGS := -frtti -std=c++11
APP_OPTIM:=release

#
APP_SHORT_COMMANDS := true

#
APP_PLATFORM            :=  android-9
#APP_CPPFLAGS            += -fexceptions
