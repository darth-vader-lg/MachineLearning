using System;
using System.Diagnostics;

namespace MachineLearning.Util
{
   /// <summary>
   /// Disposale object extensions
   /// </summary>
   internal static class DisposableExtensions
   {
      /// <summary>
      /// Dispose an object in a safely way
      /// </summary>
      /// <param name="obj">Disposable object</param>
      public static T SafeDispose<T>(this T obj) where T : IDisposable
      {
         try {
            obj?.Dispose();
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
         return obj;
      }
   }
}
