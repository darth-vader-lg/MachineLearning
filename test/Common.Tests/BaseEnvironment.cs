using LibGit2Sharp;
using System;
using System.Diagnostics;
using System.IO;
using Xunit.Abstractions;

namespace Common.Tests
{
   public partial class BaseEnvironment : ITestOutputHelper
   {
      #region Fields
      /// <summary>
      /// Helper for the test output messages
      /// </summary>
      private readonly ITestOutputHelper output;
      #endregion
      #region Properties
      /// <summary>
      /// Root path of the folder containing test data
      /// </summary>
      static protected string DataFolder => Path.GetFullPath(Path.Combine(ProjectInfo.ProjectPath, "..", "Data"));
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="output">Optional output interface</param>
      public BaseEnvironment(ITestOutputHelper output = null) => this.output = output ?? this;
      /// <summary>
      /// Clone the test assets repository
      /// </summary>
      /// <param name="repoPath">The path of the test assets repository</param>
      /// <returns>The path of the assets repository</returns>
      public string CloneAssets(string repoPath = default)
      {
         repoPath ??= Path.Combine(DataFolder, "TestAssets");
         for (var i = 0; i < 2; i++) {
            try {
               if (!Directory.Exists(repoPath)) {
                  var dir = Repository.Clone(@"https://github.com/darth-vader-lg/MachineLearning-TestAssets.git", repoPath, new CloneOptions()
                  {
                     BranchName = "main",
                     Checkout = true,
                     RecurseSubmodules = true,
                  });
               }
               var repo = new Repository(repoPath);
               return repo.Info.Path;
            }
            catch (Exception exc) {
               if (Directory.Exists(repoPath)) {
                  WriteLine(exc.Message);
                  var gitDirPath = Path.Combine(repoPath, ".git");
                  if (Directory.Exists(gitDirPath)) {
                     var dirInfo = new DirectoryInfo(gitDirPath);
                     dirInfo.Attributes &= ~(FileAttributes.ReadOnly | FileAttributes.Hidden);
                     foreach (var info in dirInfo.GetFileSystemInfos("*", SearchOption.AllDirectories))
                        info.Attributes = FileAttributes.Normal;
                  }
                  Directory.Delete(repoPath, true);
               }
            }
         }
         return null;
      }
      /// <summary>
      /// Dispose an object and make null its reference
      /// </summary>
      /// <param name="obj">Disposable object</param>
      public static void DisposeAndNullify<T>(ref T obj) where T : IDisposable
      {
         obj?.Dispose();
         obj = default;
      }
      /// <summary>
      /// Do a save action without exception consequence
      /// </summary>
      /// <param name="action">Action to do</param>
      public static void SafeAction(Action action)
      {
         try {
            action();
         }
         catch (Exception exc) {
            try {
               Debug.WriteLine(exc);
            }
            catch {
            }
         }
      }
      /// <summary>
      /// Write the output to the tracer
      /// </summary>
      /// <param name="message">Message to output</param>
      public void WriteLine(string message)
      {
         if (output != this) {
            output.WriteLine(message);
            Debug.WriteLine(message);
         }
         else
            Trace.WriteLine(message);
      }
      /// <summary>
      /// Write the output to the tracer
      /// </summary>
      /// <param name="format">Format string</param>
      /// <param name="args">Arguments</param>
      void ITestOutputHelper.WriteLine(string format, params object[] args)
      {
         if (output != this) {
            output.WriteLine(format, args);
            Debug.WriteLine(format, args);
         }
         else
            Trace.WriteLine(string.Format(format, args));
      }
      #endregion
   }
}
