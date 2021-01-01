// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
   internal static class RowCursorUtils
   {
      /// <summary>
      /// This is an error message meant to be used in the situation where a user calls a delegate as returned from
      /// <see cref="DataViewRow.GetIdGetter"/> or <see cref="DataViewRow.GetGetter{TValue}(DataViewSchema.Column)"/>.
      /// </summary>
      internal const string FetchValueStateError = "Values cannot be fetched at this time. This method was called either before the first call to "
          + nameof(DataViewRowCursor.MoveNext) + ", or at any point after that method returned false.";
   }
}
