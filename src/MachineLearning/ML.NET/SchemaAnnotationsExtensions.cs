using System;
using System.Linq;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Schema annotations extensions
   /// </summary>
   public static class SchemaAnnotationsExtensions
   {
      #region Methods
      /// <summary>
      /// Get the array of slot names of a column
      /// </summary>
      /// <param name="column">The column</param>
      /// <returns>The array of slot names</returns>
      public static string[] GetSlotNames(this DataViewSchema.Column column)
      {
         var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
         column.GetSlotNames(ref slotNames);
         return slotNames.GetValues().ToArray().Select(s => s.ToString()).ToArray();
      }
      /// <summary>
      /// Get the array of key values of a column
      /// </summary>
      /// <param name="column">The column</param>
      /// <returns>The array of key values</returns>
      public static string[] GetKeyNames(this DataViewSchema.Column column)
      {
         var keyNames = default(VBuffer<ReadOnlyMemory<char>>);
         column.GetKeyValues(ref keyNames);
         return keyNames.GetValues().ToArray().Select(s => s.ToString()).ToArray();
      }
      #endregion
   }
}
