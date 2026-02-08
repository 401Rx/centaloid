//
//  CentiloidPluginFilter.m
//  CentiloidPlugin
//
//  Centiloid Calculator for Amyloid PET - OsiriX Plugin
//
//  This plugin exports selected PET DICOM series, runs the Python
//  Centiloid computation, and imports the results back into OsiriX.
//

#import "CentiloidPluginFilter.h"
#import <OsiriXAPI/DicomStudy.h>
#import <OsiriXAPI/DicomSeries.h>
#import <OsiriXAPI/DicomImage.h>
#import <OsiriXAPI/BrowserController.h>
#import <OsiriXAPI/DicomDatabase.h>
#import <OsiriXAPI/ViewerController.h>
#import <OsiriXAPI/DCMPix.h>
#import <OsiriXAPI/Notifications.h>

@implementation CentiloidPluginFilter

- (void)initPlugin
{
    NSLog(@"CentiloidPlugin: Initialized");
}

- (long)filterImage:(NSString *)menuName
{
    NSLog(@"CentiloidPlugin: filterImage called with menu: %@", menuName);

    // Get the current browser controller
    BrowserController *browser = [BrowserController currentBrowser];
    if (!browser) {
        [self showAlert:@"Error" message:@"No browser window found."];
        return -1;
    }

    // Get selected series
    NSArray *selectedSeries = [browser databaseSelection];
    if (!selectedSeries || [selectedSeries count] == 0) {
        [self showAlert:@"No Selection"
              message:@"Please select a PET series in the database."];
        return -1;
    }

    // Find the first series object
    DicomSeries *series = nil;
    for (id obj in selectedSeries) {
        if ([obj isKindOfClass:[DicomSeries class]]) {
            series = (DicomSeries *)obj;
            break;
        } else if ([obj isKindOfClass:[DicomStudy class]]) {
            DicomStudy *study = (DicomStudy *)obj;
            NSSet *seriesSet = [study series];
            if ([seriesSet count] > 0) {
                series = [seriesSet anyObject];
            }
            break;
        }
    }

    if (!series) {
        [self showAlert:@"No Series"
              message:@"Could not find a DICOM series in selection."];
        return -1;
    }

    // Run the computation in background
    [self runCentiloidComputationForSeries:series];

    return 0;
}

- (void)runCentiloidComputationForSeries:(DicomSeries *)series
{
    // Show progress
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter]
            postNotificationName:OsirixAddToDBNotification
            object:nil
            userInfo:@{@"loading": @YES}];
    });

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            // Create temporary directory for export
            NSString *tempDir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:[[NSUUID UUID] UUIDString]];
            NSString *inputDir = [tempDir stringByAppendingPathComponent:@"input"];
            NSString *outputDir = [tempDir stringByAppendingPathComponent:@"output"];

            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:inputDir
                withIntermediateDirectories:YES
                attributes:nil
                error:nil];
            [fm createDirectoryAtPath:outputDir
                withIntermediateDirectories:YES
                attributes:nil
                error:nil];

            // Export DICOM files
            NSSet *images = [series images];
            NSLog(@"CentiloidPlugin: Exporting %lu DICOM files", (unsigned long)[images count]);

            for (DicomImage *image in images) {
                NSString *srcPath = [image completePath];
                NSString *dstPath = [inputDir stringByAppendingPathComponent:
                    [srcPath lastPathComponent]];
                [fm copyItemAtPath:srcPath toPath:dstPath error:nil];
            }

            // Run Python Centiloid computation
            NSString *pythonPath = @"/usr/bin/python3";
            NSString *scriptArgs = [NSString stringWithFormat:
                @"-m centaloid.main run \"%@\" --output \"%@\" --dicom-sc --dicom-sr",
                inputDir, outputDir];

            NSTask *task = [[NSTask alloc] init];
            [task setLaunchPath:pythonPath];
            [task setArguments:@[@"-c",
                [NSString stringWithFormat:
                    @"import sys; sys.argv = ['centaloid', 'run', '%@', '--output', '%@', '--dicom-sc', '--dicom-sr']; "
                    @"from centaloid.main import main; main()",
                    inputDir, outputDir]]];

            NSPipe *outputPipe = [NSPipe pipe];
            NSPipe *errorPipe = [NSPipe pipe];
            [task setStandardOutput:outputPipe];
            [task setStandardError:errorPipe];

            // Set environment for Python
            NSMutableDictionary *env = [[[NSProcessInfo processInfo] environment] mutableCopy];
            [env setObject:@"/usr/local/lib/python3/site-packages" forKey:@"PYTHONPATH"];
            [task setEnvironment:env];

            NSLog(@"CentiloidPlugin: Running Python computation...");

            @try {
                [task launch];
                [task waitUntilExit];
            } @catch (NSException *e) {
                NSLog(@"CentiloidPlugin: Task launch failed: %@", e);
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self showAlert:@"Error"
                          message:@"Failed to run Python. Make sure centaloid is installed."];
                });
                return;
            }

            int status = [task terminationStatus];
            NSLog(@"CentiloidPlugin: Python task finished with status %d", status);

            // Read output
            NSData *outputData = [[outputPipe fileHandleForReading] readDataToEndOfFile];
            NSString *output = [[NSString alloc] initWithData:outputData
                                                     encoding:NSUTF8StringEncoding];
            NSLog(@"CentiloidPlugin: Output: %@", output);

            if (status != 0) {
                NSData *errorData = [[errorPipe fileHandleForReading] readDataToEndOfFile];
                NSString *errorStr = [[NSString alloc] initWithData:errorData
                                                           encoding:NSUTF8StringEncoding];
                NSLog(@"CentiloidPlugin: Error: %@", errorStr);

                dispatch_async(dispatch_get_main_queue(), ^{
                    [self showAlert:@"Computation Failed"
                          message:[NSString stringWithFormat:@"Error: %@", errorStr]];
                });
                return;
            }

            // Find generated DICOM files
            NSArray *outputFiles = [fm contentsOfDirectoryAtPath:outputDir error:nil];
            NSMutableArray *dicomFiles = [NSMutableArray array];

            for (NSString *filename in outputFiles) {
                if ([filename hasSuffix:@".dcm"]) {
                    [dicomFiles addObject:[outputDir stringByAppendingPathComponent:filename]];
                }
            }

            NSLog(@"CentiloidPlugin: Found %lu DICOM output files", (unsigned long)[dicomFiles count]);

            // Import DICOM files back into OsiriX
            if ([dicomFiles count] > 0) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    BrowserController *browser = [BrowserController currentBrowser];
                    if (browser) {
                        // Add files to database
                        [browser addFilesToDatabase:dicomFiles];

                        // Refresh the browser
                        [browser refreshDatabase:nil];

                        [self showAlert:@"Centiloid Complete"
                              message:[NSString stringWithFormat:
                                  @"Computation complete. %lu result files imported.",
                                  (unsigned long)[dicomFiles count]]];
                    }
                });
            } else {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self showAlert:@"Warning"
                          message:@"Computation finished but no DICOM files were generated."];
                });
            }

            // Cleanup temp directory
            [fm removeItemAtPath:tempDir error:nil];

        } @catch (NSException *exception) {
            NSLog(@"CentiloidPlugin: Exception: %@", exception);
            dispatch_async(dispatch_get_main_queue(), ^{
                [self showAlert:@"Error"
                      message:[NSString stringWithFormat:@"Exception: %@", exception.reason]];
            });
        }
    });
}

- (void)showAlert:(NSString *)title message:(NSString *)message
{
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:title];
    [alert setInformativeText:message];
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

- (ViewerController *)duplicateCurrent2DViewerWindow
{
    return nil;
}

@end
