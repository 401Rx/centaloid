//
//  CentiloidPluginFilter.h
//  CentiloidPlugin
//
//  Centiloid Calculator for Amyloid PET - OsiriX Plugin
//

#import <Foundation/Foundation.h>
#import <OsiriXAPI/PluginFilter.h>

@interface CentiloidPluginFilter : PluginFilter

- (long)filterImage:(NSString *)menuName;
- (ViewerController *)duplicateCurrent2DViewerWindow;

@end
