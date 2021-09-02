using System.Threading.Tasks;

namespace MachineLearning.Util
{
   internal static class TaskExtensions
   {
      #region Methods
      /// <summary>
      /// Attende un task in maniera sincrona
      /// </summary>
      /// <param name="task">Task da attendere</param>
      public static void WaitSync(this Task task) => task.ConfigureAwait(false).GetAwaiter().GetResult();
      /// <summary>
      /// Attende un task in maniera sincrona
      /// </summary>
      /// <param name="task">Task da attendere</param>
      /// <returns>Il risultato del task</returns>
      public static T WaitSync<T>(this Task<T> task) => task.ConfigureAwait(false).GetAwaiter().GetResult();
      #endregion
   }
}
